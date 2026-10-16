# Phase 4 supplementary (post-hoc) computations

## 1. Throughput by cell (streams/s; per-block mean, min, max; within arrival mode only)

| Arrival | Policy | Mean | Min | Max | Blocks |
|---|---|---:|---:|---:|---:|
| sync | base | 0.8414 | 0.8402 | 0.8426 | 8 |
| sync | prio | 0.8439 | 0.8408 | 0.8455 | 8 |
| sync | both | 0.8525 | 0.8506 | 0.8542 | 8 |
| stagger | base | 0.6690 | 0.6681 | 0.6697 | 8 |
| stagger | prio | 0.6699 | 0.6693 | 0.6704 | 8 |
| stagger | both | 0.6693 | 0.6689 | 0.6702 | 8 |

## 2. Streams with an unfinished own partial at end, per block (B1-B4 round 1, then round 2)

| Arrival | Policy | Per block | Total | Files |
|---|---|---|---:|---|
| sync | base | [2, 2, 2, 2, 2, 2, 2, 2] | 16 | {'1272-135031-0001': 8, '1272-135031-0002': 8} |
| sync | prio | [4, 4, 4, 4, 4, 4, 4, 4] | 32 | {'1272-135031-0001': 16, '1272-135031-0002': 16} |
| sync | both | [4, 4, 4, 4, 4, 4, 4, 4] | 32 | {'1272-135031-0001': 16, '1272-135031-0002': 16} |
| stagger | base | [0, 0, 0, 0, 0, 0, 0, 0] | 0 | - |
| stagger | prio | [0, 0, 0, 0, 0, 0, 0, 0] | 0 | - |
| stagger | both | [0, 0, 0, 0, 0, 0, 0, 0] | 0 | - |

## 3. TTFP p95 per sync block (ms) and run 10's recovery block

| Policy | r1 B1 | r1 B2 | r1 B3 | r1 B4 | r2 B1 | r2 B2 | r2 B3 | r2 B4 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 1139 | 1148 | 1141 | 1119 | 1151 | 1149 | 1138 | 1167 |
| prio | 1172 | 1211 | 1171 | 1162 | 1172 | 1180 | 1161 | 1187 |
| both | 1173 | 1174 | 1188 | 1166 | 1159 | 1177 | 1166 | 1285 |

Run 10 recovery block: TTFP p95 1285 ms, median 954 ms. Three slowest first partials:

| Stream | File | TTFP (ms) | First-partial queue wait (ms) | First-partial decode (ms) | First frame, s after block start |
|---:|---|---:|---:|---:|---:|
| 7 | 1272-135031-0002 | 1161 | 227 | 32 | 0.01 |
| 11 | 1272-135031-0001 | 1283 | 348 | 32 | 9.35 |
| 10 | 1272-135031-0000 | 1330 | 382 | 44 | 9.27 |
