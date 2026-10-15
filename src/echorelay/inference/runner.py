"""Runs blocking inference off the event loop with bounded concurrency.

* A dedicated thread pool sized to ``max_concurrent_inference`` executes
  decodes; the event loop only awaits futures.
* A semaphore bounds in-flight jobs. The slot is released when the worker
  thread actually finishes, not when the caller stops waiting. A timed-out
  decode therefore keeps its slot until it completes, so timeouts can never
  oversubscribe the model.
* The timeout covers queueing for a slot plus the decode itself.
"""

from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .base import InferenceError, Transcriber


class InferenceTimeout(Exception):
    pass


@dataclass(frozen=True)
class TimedResult:
    text: str
    inference_s: float  # wall time inside the worker thread (monotonic clock)


class InferenceRunner:
    def __init__(self, transcriber: Transcriber, max_concurrency: int, timeout_s: float) -> None:
        self.transcriber = transcriber
        self.timeout_s = timeout_s
        self._max = max_concurrency
        self._pool = ThreadPoolExecutor(max_workers=max_concurrency, thread_name_prefix="infer")
        self._sem = asyncio.Semaphore(max_concurrency)
        self.in_flight = 0

    def _run(self, audio: np.ndarray, final: bool) -> TimedResult:
        t0 = time.monotonic()
        result = self.transcriber.transcribe(audio, final=final)
        return TimedResult(text=result.text, inference_s=time.monotonic() - t0)

    async def transcribe(self, audio: np.ndarray, *, final: bool) -> TimedResult:
        loop = asyncio.get_running_loop()
        fut: asyncio.Future | None = None
        try:
            async with asyncio.timeout(self.timeout_s):
                await self._sem.acquire()
                self.in_flight += 1
                fut = loop.run_in_executor(self._pool, self._run, audio, final)
                fut.add_done_callback(self._release)
                # shield: cancelling the caller must not "cancel" the executor
                # future (which would release the slot while the thread runs).
                return await asyncio.shield(fut)
        except TimeoutError as exc:
            raise InferenceTimeout(f"inference exceeded {self.timeout_s:.1f}s") from exc
        except InferenceError:
            raise
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise InferenceError(f"{type(exc).__name__}: {exc}") from exc
        finally:
            if fut is not None and not fut.done():
                # Caller gave up; consume the eventual result/exception so it
                # isn't reported as "never retrieved".
                fut.add_done_callback(lambda f: f.cancelled() or f.exception())

    def _release(self, _fut: asyncio.Future) -> None:
        self.in_flight -= 1
        self._sem.release()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
