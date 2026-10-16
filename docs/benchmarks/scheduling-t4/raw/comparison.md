# Scheduling policy comparison

Commit `53161b588a71`; run order: r1:base, r1:prio, r1:cancel, r1:both, r2:both, r2:cancel, r2:prio, r2:base

All times in ms. c=4 pools the c4 and recovery scenarios.

| Group | n | Fail | Final lat p50 / p95 / max | Final queue p50 / p95 | Wait own partial p95 / max | Final decode p50 / p95 (control) | TTFP p50 / p95 | Partial queue p95 | Throughput (streams/s) | Partials / stream | WER v2 | Final text = base |
|---|---:|---:|---|---|---|---|---|---:|---:|---:|---:|---:|
| base c=4 | 80 | 0 | 114 / 153 / 182 | 0 / 48 | 0 / 0 | 109 / 132 | 935 / 1026 | 128 | 0.455 | 8.00 | 5.31% | 100% |
| base c=8 | 40 | 0 | 121 / 562 / 574 | 15 / 331 | 22 / 224 | 108 / 140 | 948 / 1144 | 350 | 0.844 | 8.00 | 5.31% | 100% |
| both c=4 | 80 | 0 | 112 / 156 / 195 | 0 / 53 | 0 / 0 | 110 / 126 | 935 / 1033 | 146 | 0.456 | 8.00 | 5.31% | 100% |
| both c=8 | 40 | 0 | 117 / 266 / 317 | 19 / 148 | 34 / 72 | 110 / 129 | 953 / 1172 | 357 | 0.853 | 7.90 | 5.31% | 100% |
| cancel c=4 | 80 | 0 | 111 / 156 / 240 | 0 / 53 | 0 / 0 | 109 / 136 | 934 / 1033 | 129 | 0.455 | 8.00 | 5.31% | 100% |
| cancel c=8 | 40 | 0 | 132 / 475 / 508 | 26 / 342 | 0 / 1 | 108 / 132 | 968 / 1158 | 325 | 0.846 | 7.95 | 5.31% | 100% |
| prio c=4 | 80 | 0 | 113 / 143 / 169 | 0 / 45 | 0 / 0 | 110 / 131 | 934 / 1034 | 132 | 0.455 | 8.00 | 5.31% | 100% |
| prio c=8 | 40 | 0 | 123 / 390 / 518 | 26 / 268 | 269 / 394 | 108 / 132 | 1000 / 1176 | 388 | 0.846 | 8.00 | 5.31% | 100% |

## Differences in p95 vs baseline (bootstrap 95 % interval, ms)

| Comparison | Final latency | Final queue | TTFP | Final decode (control) |
|---|---|---|---|---|
| both vs base c=4 | +3 [-32, +44] | +5 [-23, +37] | +6 [-35, +44] | -6 [-18, +7] |
| both vs base c=8 | -296 [-358, -142] | -183 [-294, -119] | +29 [-29, +86] | -11 [-25, +17] |
| cancel vs base c=4 | +3 [-28, +33] | +5 [-19, +26] | +6 [-40, +42] | +4 [-13, +18] |
| cancel vs base c=8 | -87 [-179, +73] | +11 [-97, +66] | +15 [-67, +75] | -8 [-22, +53] |
| prio vs base c=4 | -10 [-32, +11] | -3 [-32, +32] | +7 [-36, +42] | -1 [-14, +12] |
| prio vs base c=8 | -173 [-307, +72] | -64 [-241, +80] | +33 [-28, +89] | -8 [-22, +18] |

## p95 per round (ms): consistency check

| Group | Round | Final latency | Final queue | TTFP |
|---|---|---:|---:|---:|
| base c=4 | 1 | 153 | 54 | 1026 |
| base c=4 | 2 | 137 | 46 | 1015 |
| base c=8 | 1 | 563 | 331 | 1144 |
| base c=8 | 2 | 560 | 310 | 1142 |
| both c=4 | 1 | 152 | 54 | 1015 |
| both c=4 | 2 | 156 | 49 | 1033 |
| both c=8 | 1 | 253 | 132 | 1169 |
| both c=8 | 2 | 268 | 149 | 1172 |
| cancel c=4 | 1 | 155 | 52 | 1014 |
| cancel c=4 | 2 | 156 | 53 | 1033 |
| cancel c=8 | 1 | 474 | 339 | 1159 |
| cancel c=8 | 2 | 475 | 342 | 1136 |
| prio c=4 | 1 | 143 | 60 | 1034 |
| prio c=4 | 2 | 133 | 42 | 1023 |
| prio c=8 | 1 | 376 | 256 | 1177 |
| prio c=8 | 2 | 391 | 269 | 1174 |

## Work done per run (server counters, whole run)

* **base**: inference.final.completed = [42, 42]; inference.partial.completed = [322, 323]; stream.partial_at_end.done = [38, 39]; stream.partial_at_end.queued = [1, 1]; stream.partial_at_end.running = [3, 2]; stream.partials_deferred = [2, 3]; stream.partials_scheduled = [322, 323]; stream.partials_sent = [322, 323]; stream.partials_skipped_duplicate = [0, 0]; stream.partials_skipped_empty = [0, 0]
* **prio**: inference.final.completed = [42, 42]; inference.partial.completed = [323, 323]; stream.partial_at_end.done = [36, 36]; stream.partial_at_end.queued = [3, 3]; stream.partial_at_end.running = [3, 3]; stream.partials_deferred = [2, 3]; stream.partials_promoted_at_end = [3, 3]; stream.partials_scheduled = [323, 323]; stream.partials_sent = [323, 323]; stream.partials_skipped_duplicate = [0, 0]; stream.partials_skipped_empty = [0, 0]
* **cancel**: inference.final.completed = [42, 42]; inference.partial.cancelled_while_queued = [1, 1]; inference.partial.completed = [321, 321]; stream.partial_at_end.done = [39, 39]; stream.partial_at_end.queued = [1, 1]; stream.partial_at_end.running = [2, 2]; stream.partials_cancelled_at_end = [1, 1]; stream.partials_deferred = [2, 2]; stream.partials_scheduled = [322, 322]; stream.partials_sent = [321, 321]; stream.partials_skipped_duplicate = [0, 0]; stream.partials_skipped_empty = [0, 0]
* **both**: inference.final.completed = [42, 42]; inference.partial.cancelled_while_queued = [2, 2]; inference.partial.completed = [320, 320]; stream.partial_at_end.done = [36, 36]; stream.partial_at_end.queued = [2, 2]; stream.partial_at_end.running = [4, 4]; stream.partials_cancelled_at_end = [2, 2]; stream.partials_deferred = [2, 2]; stream.partials_scheduled = [322, 322]; stream.partials_sent = [320, 320]; stream.partials_skipped_duplicate = [0, 0]; stream.partials_skipped_empty = [0, 0]
