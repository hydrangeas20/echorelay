# Phase 4 staggered-arrival analysis

Commit `b9e86349b6e5`; plan `phase4`; deviations from the registered plan: none.

All intervals are exploratory (see docs/experiments/phase4-staggered-arrival.md).

## Workload descriptors (mean over blocks)

| Cell | Active (client) mean / share at 8 | Active at end | Ends ±250 ms share / mean | Partial nbrs ±50 ms / share ≥3 / phase R | Server queue depth mean / share ≥1 / max | Queue depth at end |
|---|---|---|---|---|---|---|
| stagger base | 5.80 / 47% | 6.61 | 35% / 0.49 | 0.21 / 0% / 0.18 | 0.06 / 5% / 2 | 0.08 |
| stagger both | 5.80 / 48% | 6.60 | 35% / 0.49 | 0.21 / 0% / 0.20 | 0.07 / 6% / 2 | 0.12 |
| stagger prio | 5.81 / 48% | 6.61 | 35% / 0.49 | 0.23 / 0% / 0.19 | 0.07 / 6% / 2 | 0.12 |
| sync base | 7.40 / 86% | 6.64 | 80% / 0.86 | 2.76 / 45% / 0.54 | 0.76 / 28% / 7 | 1.19 |
| sync both | 7.42 / 87% | 6.67 | 70% / 0.89 | 2.68 / 43% / 0.40 | 0.75 / 25% / 7 | 1.15 |
| sync prio | 7.37 / 86% | 6.71 | 70% / 0.90 | 2.83 / 45% / 0.51 | 0.81 / 27% / 7 | 1.20 |

## Latency by cell (pooled over 8 blocks; ms)

| Cell | n | Fail | Final lat p50 / p95 / max | Final queue p95 | Wait own p95 / max | Final decode p95 | TTFP p50 / p95 | Partial queue p95 | WER v2 | Unfinished own partial at end | Mean wait own (all / unfinished) |
|---|---:|---:|---|---:|---|---:|---|---:|---:|---|---|
| stagger base | 160 | 0 | 122 / 183 / 226 | 82 | 0 / 2 | 142 | 937 / 989 | 55 | 5.31% | 0.0% ({'done': 160}) | 0 / 0 |
| stagger both | 160 | 0 | 118 / 164 / 233 | 77 | 0 / 2 | 141 | 938 / 974 | 62 | 5.31% | 0.0% ({'done': 160}) | 0 / 0 |
| stagger prio | 160 | 0 | 118 / 178 / 225 | 77 | 0 / 1 | 141 | 936 / 970 | 61 | 5.31% | 0.0% ({'done': 160}) | 0 / 0 |
| sync base | 160 | 0 | 128 / 619 / 651 | 390 | 60 / 255 | 137 | 960 / 1151 | 358 | 5.31% | 10.0% ({'done': 144, 'queued': 8, 'running': 8}) | 13 / 128 |
| sync both | 160 | 0 | 122 / 291 / 353 | 165 | 61 / 98 | 137 | 961 / 1175 | 368 | 5.31% | 20.0% ({'done': 128, 'running': 14, 'queued': 18}) | 6 / 31 |
| sync prio | 160 | 0 | 132 / 419 / 637 | 282 | 283 / 499 | 136 | 1007 / 1179 | 430 | 5.31% | 20.0% ({'done': 128, 'running': 8, 'queued': 24}) | 47 / 236 |

## Final latency p95 per block (ms)

| Cell | r1 B1 | r1 B2 | r1 B3 | r1 B4 | r2 B1 | r2 B2 | r2 B3 | r2 B4 | per round |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stagger base | 193 | 168 | 176 | 149 | 161 | 164 | 174 | 177 | 184 / 174 |
| stagger both | 164 | 162 | 174 | 166 | 179 | 163 | 153 | 161 | 174 / 163 |
| stagger prio | 176 | 178 | 193 | 173 | 171 | 176 | 165 | 164 | 182 / 175 |
| sync base | 615 | 590 | 624 | 594 | 567 | 636 | 619 | 608 | 615 / 619 |
| sync both | 288 | 268 | 263 | 294 | 273 | 286 | 282 | 267 | 291 / 286 |
| sync prio | 390 | 426 | 398 | 383 | 400 | 370 | 380 | 400 | 420 / 399 |

## Comparisons (within arrival mode)

### stagger: both vs base

**Outcome: benefit not demonstrated under this workload.** Failed: C1 block-bootstrap interval for final latency p95 difference entirely below 0.

| Metric | Δ p95 (ms) | block bootstrap 95 % | stream bootstrap 95 % |
|---|---:|---|---|
| final_latency_s | -20 | [-28, +0] | [-29, +6] |
| final_queue_s | -5 | [-25, +5] | [-28, +9] |
| wait_own_partial_s | +0 | [-0, +0] | [-0, +0] |
| final_decode_s | -1 | [-11, +11] | [-10, +10] |
| ttfp_s | -14 | [-22, +11] | [-26, +14] |

Paired blocks (same round and position) with lower candidate p95: 6 of 8.

Own-partial mechanism (candidate − baseline): unfinished share +0.0pp [+0.0, +0.0]; queued share +0.0pp [+0.0, +0.0]; mean wait own -0.0 ms [-0.0, +0.0]. (Descriptive; the registered mechanism test is prio vs base.)

### stagger: prio vs base

**Outcome: benefit not demonstrated under this workload.** Failed: C1 block-bootstrap interval for final latency p95 difference entirely below 0; C2 candidate final latency p95 below baseline in every round.

| Metric | Δ p95 (ms) | block bootstrap 95 % | stream bootstrap 95 % |
|---|---:|---|---|
| final_latency_s | -5 | [-15, +15] | [-21, +24] |
| final_queue_s | -5 | [-26, +3] | [-31, +8] |
| wait_own_partial_s | +0 | [-0, +0] | [-0, +0] |
| final_decode_s | -0 | [-8, +5] | [-7, +5] |
| ttfp_s | -19 | [-26, +10] | [-29, +15] |

Paired blocks (same round and position) with lower candidate p95: 3 of 8.

Own-partial mechanism (candidate − baseline): unfinished share +0.0pp [+0.0, +0.0]; queued share +0.0pp [+0.0, +0.0]; mean wait own -0.0 ms [-0.0, +0.0]. Registered mechanism test: frequency increase supported: False; duration increase supported: False.

### sync: both vs base

**Outcome: benefit demonstrated under this workload.**

| Metric | Δ p95 (ms) | block bootstrap 95 % | stream bootstrap 95 % |
|---|---:|---|---|
| final_latency_s | -328 | [-344, -302] | [-360, -273] |
| final_queue_s | -225 | [-230, -210] | [-308, -176] |
| wait_own_partial_s | +1 | [-4, +22] | [-184, +60] |
| final_decode_s | -1 | [-14, +11] | [-15, +13] |
| ttfp_s | +24 | [+4, +56] | [-0, +57] |

Paired blocks (same round and position) with lower candidate p95: 8 of 8.

Own-partial mechanism (candidate − baseline): unfinished share +10.0pp [+10.0, +10.0]; queued share +6.2pp [+5.0, +8.1]; mean wait own -6.8 ms [-8.1, -5.5]. (Descriptive; the registered mechanism test is prio vs base.)

### sync: prio vs base

**Outcome: benefit demonstrated under this workload.**

| Metric | Δ p95 (ms) | block bootstrap 95 % | stream bootstrap 95 % |
|---|---:|---|---|
| final_latency_s | -200 | [-228, -174] | [-247, -67] |
| final_queue_s | -108 | [-118, -94] | [-196, +36] |
| wait_own_partial_s | +223 | [+214, +244] | [+38, +383] |
| final_decode_s | -1 | [-9, +8] | [-13, +10] |
| ttfp_s | +28 | [+6, +50] | [+3, +52] |

Paired blocks (same round and position) with lower candidate p95: 8 of 8.

Own-partial mechanism (candidate − baseline): unfinished share +10.0pp [+10.0, +10.0]; queued share +10.0pp [+10.0, +10.0]; mean wait own +34.3 ms [+32.0, +37.2]. Registered mechanism test: frequency increase supported: True; duration increase supported: True.

