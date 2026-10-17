"""Inference slot scheduler.

Replaces a plain ``asyncio.Semaphore`` so that (a) waiting requests can be
ordered by priority, (b) a waiting request can be promoted, and (c) callers
can see whether their request is still queued, which safe cancellation needs.

Concurrency model: every method runs on the event loop thread. Worker
threads never touch the scheduler; the runner releases slots from a future
done-callback, which asyncio also runs on the loop. No locks are needed, and
no ``await`` separates a check from the action that depends on it.

Invariants (tested in tests/test_scheduler.py):

S1 Conservation: ``free + held == slots`` at every point between callbacks;
   a slot is never double-granted and never lost, including when a granted
   waiter is cancelled before it resumes.
S2 No barging: a new request takes a free slot immediately only when nobody
   is waiting; otherwise it queues. A slot freed by ``release`` is handed
   directly to the next waiter rather than returned to the pool, so a free
   slot and a live waiter never coexist (the ``has_waiters`` check in
   ``acquire`` is therefore defensive; the invariant itself is tested).
S3 Order: waiters are served by (priority, arrival). With every request at
   the same priority, as when final priority is disabled, this is plain FIFO,
   the same order an ``asyncio.Semaphore`` gives.
S4 Cancellation: cancelling a request that is still queued never consumes a
   slot. If cancellation races with a grant, the granted slot is passed on.
"""

from __future__ import annotations

import asyncio
import heapq
import itertools
from dataclasses import dataclass, field

# Lower value = served first.
PRIORITY_FINAL = 0
PRIORITY_PARTIAL = 1


@dataclass
class Job:
    """Handle for one inference request; state is only mutated on the loop.

    created -> queued -> running -> done
    created -> running -> done          (slot was free, no waiters)
    queued  -> cancelled                (cancelled before it got a slot)
    """

    kind: str  # "partial" | "final"
    priority: int
    state: str = "created"
    promoted: bool = False
    _fut: asyncio.Future | None = field(default=None, repr=False)
    _entry: list | None = field(default=None, repr=False)

    @property
    def queued(self) -> bool:
        return self.state == "queued"


class SlotScheduler:
    def __init__(self, slots: int, prioritize: bool = False) -> None:
        self.slots = slots
        self.prioritize = prioritize
        self.free = slots
        self._heap: list[list] = []  # entries: [priority, seq, job, valid]
        self._seq = itertools.count()
        self.waiting = 0
        self.granted_total = 0
        self.handoffs_after_cancel = 0  # S4 race resolved (granted, then cancelled)

    # ------------------------------------------------------------- helpers
    def _effective(self, priority: int) -> int:
        return priority if self.prioritize else PRIORITY_PARTIAL

    def _push(self, job: Job, priority: int) -> None:
        entry = [self._effective(priority), next(self._seq), job, True]
        job._entry = entry
        heapq.heappush(self._heap, entry)

    def _prune(self) -> None:
        while self._heap and (not self._heap[0][3] or self._heap[0][2]._fut.done()):
            heapq.heappop(self._heap)

    def has_waiters(self) -> bool:
        self._prune()
        return bool(self._heap)

    # -------------------------------------------------------------- public
    async def acquire(self, job: Job) -> None:
        """Wait for a slot. On return the caller holds one and must
        ``release()`` it exactly once."""
        if self.free > 0 and not self.has_waiters():  # S2
            self.free -= 1
            self.granted_total += 1
            job.state = "running"
            return
        job._fut = asyncio.get_running_loop().create_future()
        job.state = "queued"
        self._push(job, job.priority)
        self.waiting += 1
        try:
            await job._fut
        except asyncio.CancelledError:
            if job._fut.done() and not job._fut.cancelled():
                # S4: the slot was handed to us, but we were cancelled before
                # resuming. Pass it on rather than leaking it.
                self.handoffs_after_cancel += 1
                self.release()
            job.state = "cancelled"
            raise
        finally:
            self.waiting -= 1
            if job._entry is not None:
                job._entry[3] = False
        job.state = "running"

    def release(self) -> None:
        """Give one held slot to the best waiter, or back to the pool."""
        while self._heap:
            entry = heapq.heappop(self._heap)
            _, _, job, valid = entry
            if not valid or job._fut.done():
                continue  # cancelled or superseded by a promotion
            entry[3] = False
            self.granted_total += 1
            job._fut.set_result(None)  # S2: direct hand-off; ``free`` unchanged
            return
        self.free += 1
        if self.free > self.slots:  # S1 guard: would mean a double release
            raise RuntimeError("slot released more times than acquired")

    def promote(self, job: Job, priority: int) -> bool:
        """Raise the priority of a job that has not started (priority
        inheritance). Returns whether anything changed.

        * ``created`` (its task has not reached ``acquire`` yet): the job will
          queue at the new priority.
        * ``queued``: re-queued at the new priority; its arrival order within
          the new class is the time of promotion.
        * running / done / cancelled: nothing to do.
        """
        if not self.prioritize or priority >= job.priority:
            return False
        if job.state == "created":
            job.priority = priority
            job.promoted = True
            return True
        if job.state != "queued" or job._entry is None:
            return False
        job.priority = priority
        job._entry[3] = False  # lazy-delete the old entry
        self._push(job, priority)
        job.promoted = True
        return True
