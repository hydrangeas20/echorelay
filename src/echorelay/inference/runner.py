"""Runs blocking inference off the event loop with bounded concurrency.

* A dedicated thread pool sized to ``max_concurrent_inference`` executes
  decodes; the event loop only awaits futures.
* A :class:`SlotScheduler` bounds in-flight jobs. The slot is released when
  the worker thread actually finishes, not when the caller stops waiting. A
  timed-out decode therefore keeps its slot until it completes, so timeouts
  can never oversubscribe the model.
* Waiting requests are served in arrival order (FIFO). With
  ``final_priority`` enabled, finals are served before partials.
* The timeout covers queueing for a slot plus the decode itself.
* Queue wait (call -> slot acquired) and decode time (inside the worker
  thread) are measured separately, so latency can be decomposed.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from ..config import SAMPLE_RATE_HZ
from ..metrics import LevelTracker, Metrics
from .base import InferenceError, Transcriber
from .scheduler import PRIORITY_FINAL, PRIORITY_PARTIAL, Job, SlotScheduler


class InferenceTimeout(Exception):
    pass


@dataclass(frozen=True)
class TimedResult:
    text: str
    inference_s: float  # wall time inside the worker thread (monotonic clock)
    queue_s: float = 0.0  # call -> inference slot acquired (monotonic clock)


class InferenceRunner:
    def __init__(
        self,
        transcriber: Transcriber,
        max_concurrency: int,
        timeout_s: float,
        metrics: Metrics | None = None,
        final_priority: bool = False,
    ) -> None:
        self.transcriber = transcriber
        self.timeout_s = timeout_s
        self.max_concurrency = max_concurrency
        self.metrics = metrics or Metrics()
        self._pool = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="infer")
        self.scheduler = SlotScheduler(max_concurrency, prioritize=final_priority)
        self.in_flight = 0  # jobs holding a slot (running in a worker thread)
        self.in_flight_level = LevelTracker()  # time at each in-flight count

    @property
    def waiting(self) -> int:
        """Callers waiting for a slot."""
        return self.scheduler.waiting

    @staticmethod
    def new_job(final: bool) -> Job:
        return Job("final" if final else "partial", PRIORITY_FINAL if final else PRIORITY_PARTIAL)

    def _run(self, audio: np.ndarray, final: bool) -> tuple[str, float]:
        t0 = time.monotonic()
        result = self.transcriber.transcribe(audio, final=final)
        return result.text, time.monotonic() - t0

    async def transcribe(self, audio: np.ndarray, *, final: bool, job: Job | None = None) -> TimedResult:
        """Decode ``audio``. Pass a ``job`` (from :meth:`new_job`) to observe
        whether the request is still queued, e.g. to cancel it safely."""
        loop = asyncio.get_running_loop()
        job = job or self.new_job(final)
        kind = job.kind
        m = self.metrics
        fut: asyncio.Future | None = None
        t_call = time.monotonic()
        queue_s: float | None = None
        try:
            async with asyncio.timeout(self.timeout_s):
                await self.scheduler.acquire(job)
                queue_s = time.monotonic() - t_call
                self.in_flight += 1
                self.in_flight_level.set(self.in_flight)
                fut = loop.run_in_executor(self._pool, self._run, audio, final)
                fut.add_done_callback(lambda _f, j=job: self._release(j))
                # shield: cancelling the caller must not "cancel" the executor
                # future (which would release the slot while the thread runs).
                text, inference_s = await asyncio.shield(fut)
        except TimeoutError as exc:
            m.inc(f"inference.{kind}.timeouts")
            if queue_s is None:
                m.inc(f"inference.{kind}.timeouts_while_queued")
            raise InferenceTimeout(f"inference exceeded {self.timeout_s:.1f}s") from exc
        except InferenceError:
            m.inc(f"inference.{kind}.errors")
            raise
        except asyncio.CancelledError:
            m.inc(f"inference.{kind}.cancelled_callers")
            if queue_s is None:
                m.inc(f"inference.{kind}.cancelled_while_queued")
            raise
        except Exception as exc:
            m.inc(f"inference.{kind}.errors")
            raise InferenceError(f"{type(exc).__name__}: {exc}") from exc
        else:
            audio_s = audio.size / SAMPLE_RATE_HZ
            m.inc(f"inference.{kind}.completed")
            m.observe(f"inference.{kind}.queue_wait_s", queue_s)
            m.observe(f"inference.{kind}.decode_s", inference_s)
            m.observe(f"inference.{kind}.audio_s", audio_s)
            if audio_s > 0:
                m.observe(f"inference.{kind}.rtf", inference_s / audio_s)
            return TimedResult(text=text, inference_s=inference_s, queue_s=queue_s)
        finally:
            if fut is not None and not fut.done():
                # Caller gave up; consume the eventual result/exception so it
                # isn't reported as "never retrieved".
                fut.add_done_callback(lambda f: f.cancelled() or f.exception())

    def _release(self, job: Job) -> None:
        # Runs on the loop (future done-callback) once the worker thread is done.
        job.state = "done"
        self.in_flight -= 1
        self.in_flight_level.set(self.in_flight)
        self.scheduler.release()

    def snapshot(self) -> dict:
        return {
            "in_flight": self.in_flight,
            "waiting": self.waiting,
            "max_concurrent": self.max_concurrency,
        }

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
