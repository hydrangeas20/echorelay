"""In-process metrics: counters and bounded-window distributions.

No external dependencies or infrastructure. Exposed as JSON at ``/metrics``.
Definitions of every metric are in docs/measurements.md.

Memory is bounded: each distribution keeps only its most recent
``window`` observations (default 10,000) plus a lifetime count and max.
Percentiles are computed over that window.
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import threading
import time
from collections import Counter, deque
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class Distribution:
    __slots__ = ("_window", "count", "max", "total")

    def __init__(self, window: int) -> None:
        self._window: deque[float] = deque(maxlen=window)
        self.count = 0
        self.max = float("-inf")
        self.total = 0.0

    def observe(self, value: float) -> None:
        self._window.append(value)
        self.count += 1
        self.total += value
        if value > self.max:
            self.max = value

    def summary(self) -> dict[str, Any]:
        if not self.count:
            return {"count": 0}
        a = np.fromiter(self._window, dtype=float)
        p50, p95, p99 = np.percentile(a, [50, 95, 99])
        return {
            "count": self.count,  # lifetime observations
            "window": int(a.size),  # observations the percentiles are computed over
            "mean": float(a.mean()),
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
            "max": self.max,  # lifetime max
            "sum": self.total,  # lifetime sum
        }


class Metrics:
    """Thread-safe registry. Observations come from the event loop and
    from inference worker callbacks, so all mutation takes a lock."""

    def __init__(self, window: int = 10_000) -> None:
        self._window = window
        self._lock = threading.Lock()
        self._dists: dict[str, Distribution] = {}
        self._counters: Counter[str] = Counter()
        self.started = time.monotonic()

    def observe(self, name: str, value: float) -> None:
        with self._lock:
            d = self._dists.get(name)
            if d is None:
                d = self._dists[name] = Distribution(self._window)
            d.observe(value)

    def inc(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._counters[name] += n

    def counter(self, name: str) -> int:
        with self._lock:
            return self._counters[name]

    def summary(self, name: str) -> dict[str, Any]:
        with self._lock:
            d = self._dists.get(name)
            return d.summary() if d else {"count": 0}

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "uptime_s": time.monotonic() - self.started,
                "counters": dict(sorted(self._counters.items())),
                "distributions": {k: d.summary() for k, d in sorted(self._dists.items())},
            }


def process_resources() -> dict[str, Any]:
    """Process-level resource usage (Linux /proc when available)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    out: dict[str, Any] = {
        "cpu_user_s": usage.ru_utime,
        "cpu_system_s": usage.ru_stime,
        "max_rss_bytes": usage.ru_maxrss * 1024,  # Linux reports KiB
        "threads": threading.active_count(),
    }
    try:
        with open("/proc/self/statm") as f:
            rss_pages = int(f.read().split()[1])
        out["rss_bytes"] = rss_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        pass
    return out


async def monitor_event_loop_lag(metrics: Metrics, interval_s: float = 0.1) -> None:
    """Measure how late the event loop wakes a sleeping task.

    lag = actual wake time - scheduled wake time. Sustained lag well above a
    few milliseconds means something is blocking the loop (e.g. inference
    accidentally running on it).
    """
    loop = asyncio.get_running_loop()
    while True:
        expected = loop.time() + interval_s
        await asyncio.sleep(interval_s)
        metrics.observe("event_loop.lag_s", max(0.0, loop.time() - expected))
