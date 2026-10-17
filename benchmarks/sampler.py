"""Background resource sampling during a load test.

Every ``interval_s`` the sampler records, with a client monotonic timestamp:

* server ``/metrics`` gauges: active streams, in-flight and waiting
  inference jobs, buffered audio bytes, process RSS / CPU / threads, and
  event-loop lag (lifetime max and windowed p99);
* GPU state from ``nvidia-smi`` when available: utilization, memory used,
  SM clock (current and max), power draw, temperature.

Both calls run off the event loop (thread / subprocess) so sampling does not
delay the streams being measured. Failures are recorded per sample, never
raised. Servers without ``/metrics`` (Phase 1) are handled.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import time
import urllib.error
import urllib.request

import numpy as np

GPU_FIELDS = ("utilization.gpu", "memory.used", "clocks.sm", "clocks.max.sm", "power.draw",
              "temperature.gpu")
GPU_KEYS = ("gpu_util_pct", "gpu_mem_used_mib", "sm_clock_mhz", "sm_clock_max_mhz", "power_w", "temp_c")


def fetch_json(url: str, timeout: float = 5.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read())
    except (urllib.error.URLError, OSError, ValueError):
        return None


def extract_server(m: dict) -> dict:
    lag = m.get("distributions", {}).get("event_loop.lag_s", {})
    return {
        "active": m["streams"]["active"],
        "buffered_audio_bytes": m["streams"].get("buffered_audio_bytes"),
        "in_flight": m["inference"]["in_flight"],
        "waiting": m["inference"].get("waiting"),
        "rss_bytes": m["process"].get("rss_bytes"),
        "cpu_s": m["process"]["cpu_user_s"] + m["process"]["cpu_system_s"],
        "threads": m["process"]["threads"],
        "loop_lag_max_s": lag.get("max"),
        "loop_lag_p99_s": lag.get("p99"),
    }


def parse_nvidia_smi(line: str) -> dict:
    out = {}
    for key, raw in zip(GPU_KEYS, line.split(",")):
        try:
            out[key] = float(raw.strip())
        except ValueError:
            out[key] = None  # "[N/A]" on some GPUs
    return out


async def query_gpu() -> list[dict] | None:
    if not shutil.which("nvidia-smi"):
        return None
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi", f"--query-gpu={','.join(GPU_FIELDS)}", "--format=csv,noheader,nounits",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
    return [parse_nvidia_smi(line) for line in stdout.decode().splitlines() if line.strip()]


async def sample(metrics_url: str, interval_s: float, stop: asyncio.Event) -> list[dict]:
    samples: list[dict] = []
    t0 = time.monotonic()
    while not stop.is_set():
        tick = time.monotonic()
        s: dict = {"t": tick - t0}
        m = await asyncio.to_thread(fetch_json, metrics_url)
        if m is not None and "streams" in m:
            s["server"] = extract_server(m)
        else:
            s["server_error"] = "metrics unavailable"
        try:
            gpus = await query_gpu()
            if gpus is not None:
                s["gpus"] = gpus
        except (OSError, asyncio.TimeoutError) as exc:
            s["gpu_error"] = type(exc).__name__
        samples.append(s)
        try:
            await asyncio.wait_for(stop.wait(), timeout=max(0.0, interval_s - (time.monotonic() - tick)))
        except asyncio.TimeoutError:
            pass
    return samples


def summarize_samples(samples: list[dict]) -> dict:
    def col(path: tuple[str, ...]) -> list[float]:
        vals = []
        for s in samples:
            v = s
            for k in path:
                v = v.get(k) if isinstance(v, dict) else None
                if v is None:
                    break
            if isinstance(v, (int, float)):
                vals.append(float(v))
        return vals

    def stats(vals: list[float]) -> dict | None:
        if not vals:
            return None
        a = np.asarray(vals)
        return {"min": float(a.min()), "p50": float(np.median(a)), "mean": float(a.mean()), "max": float(a.max())}

    out = {"num_samples": len(samples)}
    for key in ("active", "in_flight", "waiting", "buffered_audio_bytes", "rss_bytes", "threads",
                "loop_lag_max_s"):
        out[key] = stats(col(("server", key)))
    cpu = [(s["t"], s["server"]["cpu_s"]) for s in samples if "server" in s]
    if len(cpu) >= 2 and cpu[-1][0] > cpu[0][0]:
        # Rate over the samples that carry CPU readings (skips failed polls).
        out["server_cpu_cores_mean"] = (cpu[-1][1] - cpu[0][1]) / (cpu[-1][0] - cpu[0][0])
    gpu_samples = [s["gpus"][0] for s in samples if s.get("gpus")]
    for key in GPU_KEYS:
        out[key] = stats([g[key] for g in gpu_samples if g.get(key) is not None])
    return out
