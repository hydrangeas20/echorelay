"""SlotScheduler invariants (S1-S4 in scheduler.py). Fully deterministic:
no threads and no sleeps; ``asyncio.sleep(0)`` yields let woken waiters resume."""

from __future__ import annotations

import asyncio
import random

import pytest

from echorelay.inference.scheduler import PRIORITY_FINAL, PRIORITY_PARTIAL, Job, SlotScheduler


async def settle(n: int = 5) -> None:
    for _ in range(n):
        await asyncio.sleep(0)


def job(kind: str = "partial") -> Job:
    return Job(kind, PRIORITY_FINAL if kind == "final" else PRIORITY_PARTIAL)


class Harness:
    """Tasks that acquire a slot and record the order in which they got it."""

    def __init__(self, sched: SlotScheduler) -> None:
        self.sched = sched
        self.order: list[str] = []
        self.tasks: dict[str, asyncio.Task] = {}
        self.jobs: dict[str, Job] = {}

    def request(self, name: str, kind: str = "partial") -> None:
        j = self.jobs[name] = job(kind)

        async def run():
            await self.sched.acquire(j)
            self.order.append(name)

        self.tasks[name] = asyncio.create_task(run())

    def held(self) -> int:
        return self.sched.slots - self.sched.free


async def test_fifo_when_priority_disabled_even_for_finals():
    h = Harness(SlotScheduler(1, prioritize=False))
    h.request("hold")
    await settle()
    for name, kind in [("p1", "partial"), ("f1", "final"), ("p2", "partial"), ("f2", "final")]:
        h.request(name, kind)
    await settle()
    for _ in range(4):
        h.sched.release()
        await settle()
    assert h.order == ["hold", "p1", "f1", "p2", "f2"]


async def test_finals_first_then_fifo_within_class_when_enabled():
    h = Harness(SlotScheduler(1, prioritize=True))
    h.request("hold")
    await settle()
    for name, kind in [("p1", "partial"), ("f1", "final"), ("p2", "partial"), ("f2", "final")]:
        h.request(name, kind)
    await settle()
    for _ in range(4):
        h.sched.release()
        await settle()
    assert h.order == ["hold", "f1", "f2", "p1", "p2"]


async def test_no_barging_freed_slot_goes_to_waiter_not_newcomer():
    h = Harness(SlotScheduler(1, prioritize=True))
    h.request("hold")
    await settle()
    h.request("waiter", "partial")
    await settle()
    h.sched.release()  # handed to "waiter", which has not resumed yet
    h.request("newcomer", "final")  # higher priority, but must not barge in
    await settle()
    assert h.order == ["hold", "waiter"]
    assert h.sched.free == 0
    h.sched.release()
    await settle()
    assert h.order == ["hold", "waiter", "newcomer"]


async def test_cancel_while_queued_consumes_no_slot():
    h = Harness(SlotScheduler(1))
    h.request("hold")
    await settle()
    h.request("victim")
    await settle()
    assert h.jobs["victim"].queued and h.sched.waiting == 1
    h.tasks["victim"].cancel()
    await settle()
    assert h.jobs["victim"].state == "cancelled" and h.sched.waiting == 0
    h.sched.release()
    assert h.sched.free == 1  # returned to the pool, not given to the cancelled waiter


async def test_grant_then_cancel_race_passes_slot_on():
    """S4: release() hands the slot to A; A is cancelled before it resumes.
    The slot must go to B, not leak and not be double-granted."""
    h = Harness(SlotScheduler(1))
    h.request("hold")
    await settle()
    h.request("A")
    h.request("B")
    await settle()
    h.sched.release()  # A granted (future resolved), A not yet resumed
    h.tasks["A"].cancel()  # cancelled in the same loop iteration
    await settle()
    assert h.order == ["hold", "B"]
    assert h.jobs["A"].state == "cancelled"
    assert h.sched.handoffs_after_cancel == 1
    assert h.held() == 1 and h.sched.waiting == 0
    h.sched.release()
    assert h.sched.free == 1


async def test_timeout_while_queued_does_not_leak():
    s = SlotScheduler(1)
    await s.acquire(job())
    with pytest.raises(TimeoutError):
        async with asyncio.timeout(0.01):
            await s.acquire(job())
    assert s.waiting == 0
    s.release()
    assert s.free == 1
    await s.acquire(job())  # still usable
    s.release()


async def test_promote_moves_queued_partial_ahead_of_partials_not_earlier_finals():
    h = Harness(SlotScheduler(1, prioritize=True))
    h.request("hold")
    await settle()
    for name, kind in [("p1", "partial"), ("f1", "final"), ("own", "partial"), ("p2", "partial")]:
        h.request(name, kind)
    await settle()
    assert h.sched.promote(h.jobs["own"], PRIORITY_FINAL) is True
    h.request("f2", "final")  # arrives after the promotion
    await settle()
    for _ in range(5):
        h.sched.release()
        await settle()
    assert h.order == ["hold", "f1", "own", "f2", "p1", "p2"]
    assert h.jobs["own"].promoted


async def test_promote_is_a_no_op_when_disabled_or_not_queued():
    h = Harness(SlotScheduler(1, prioritize=False))
    h.request("hold")
    await settle()
    h.request("p")
    await settle()
    assert h.sched.promote(h.jobs["p"], PRIORITY_FINAL) is False
    running = h.jobs["hold"]
    s2 = SlotScheduler(1, prioritize=True)
    assert s2.promote(running, PRIORITY_FINAL) is False  # running, not queued


def test_double_release_is_detected():
    s = SlotScheduler(2)  # nothing held: any release is one too many
    with pytest.raises(RuntimeError, match="released more times"):
        s.release()


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("prioritize", [False, True])
async def test_randomized_invariants(seed, prioritize):
    """Random interleavings of requests, releases, cancellations and
    promotions. After every step: S1 conservation, and each grant goes to
    the best live waiter by (effective priority, arrival)."""
    rng = random.Random(seed)
    slots = rng.choice([1, 2, 3])
    s = SlotScheduler(slots, prioritize=prioritize)
    holding: list[str] = []
    live: dict[str, tuple] = {}  # name -> (priority, seq) for queued waiters
    order: list[str] = []
    tasks: dict[str, asyncio.Task] = {}
    jobs: dict[str, Job] = {}
    seq = 0
    violations: list[str] = []

    def eff(p):
        return p if prioritize else PRIORITY_PARTIAL

    for step in range(120):
        op = rng.random()
        if op < 0.45:
            name, kind = f"j{step}", rng.choice(["partial", "final"])
            j = jobs[name] = job(kind)
            expect_immediate = s.free > 0 and not s.has_waiters()

            async def run(n=name, jj=j):
                await s.acquire(jj)
                order.append(n)
                holding.append(n)

            tasks[name] = asyncio.create_task(run())
            if not expect_immediate:
                seq += 1
                live[name] = (eff(j.priority), seq)
        elif op < 0.75 and holding:
            holding.pop(rng.randrange(len(holding)))
            if live:
                best = min(live, key=live.get)
                s.release()
                await settle()
                if order and order[-1] != best and best in live:
                    violations.append(f"step {step}: granted {order[-1]}, best was {best}")
                live.pop(order[-1], None)
                continue
            s.release()
        elif op < 0.9 and live:
            victim = rng.choice(sorted(live))
            tasks[victim].cancel()
            live.pop(victim)
        elif live and prioritize:
            name = rng.choice(sorted(live))
            if jobs[name].kind == "partial" and live[name][0] == PRIORITY_PARTIAL:
                if s.promote(jobs[name], PRIORITY_FINAL):
                    seq += 1
                    live[name] = (PRIORITY_FINAL, seq)
        await settle()
        # S1: every slot is either free or held by a task that resumed with it
        assert s.free + len(holding) == slots, (step, s.free, holding)
        assert s.waiting == len(live), (step, s.waiting, live)
        # Hand-off invariant behind S2: a free slot never coexists with a waiter.
        assert not (s.free > 0 and s.waiting > 0), (step, s.free, s.waiting)
    assert not violations, violations[:3]
    for t in tasks.values():
        t.cancel()
    await settle()


@pytest.mark.parametrize("seed", range(10))
async def test_fifo_mode_matches_asyncio_semaphore(seed):
    """With priority disabled the scheduler grants in exactly the order an
    asyncio.Semaphore does, for the same random sequence of operations."""
    rng = random.Random(seed)
    ops = []
    for i in range(80):
        r = rng.random()
        ops.append(("req", f"j{i}") if r < 0.5 else ("rel", None) if r < 0.85 else ("cancel", rng.random()))

    async def replay(make_acquire, release):
        order, holding, tasks, queued = [], [], {}, []
        for kind, arg in ops:
            if kind == "req":
                async def run(n=arg):
                    await make_acquire()
                    order.append(n)
                    holding.append(n)
                tasks[arg] = asyncio.create_task(run())
                queued.append(arg)
            elif kind == "rel" and holding:
                holding.pop(0)
                release()
            elif kind == "cancel":
                waiting = [n for n in queued if n not in order and not tasks[n].done()]
                if waiting:
                    tasks[waiting[int(arg * len(waiting))]].cancel()
            await settle()
        for t in tasks.values():
            t.cancel()
        await settle()
        return order

    sem = asyncio.Semaphore(2)
    sched = SlotScheduler(2, prioritize=False)
    a = await replay(sem.acquire, sem.release)
    b = await replay(lambda: sched.acquire(job()), sched.release)
    assert a == b
