Scenario `c8`; latencies in ms; class = state of the stream's own partial at `end`.

| Policy | Class | n (share) | Promoted / cancelled | Wait own p50 / max | Final queue p50 / max | Final latency p50 / max | Streams ≥ policy p95 |
|---|---|---|---|---|---|---|---:|
| base (p95 562) | done | 37 (92%) | 0 / 0 | 0 / 1 | 0 / 413 | 114 / 560 | 0 |
| base (p95 562) | running | 1 (2%) | 0 / 0 | 13 / 13 | 408 / 408 | 562 / 562 | 0 |
| base (p95 562) | queued | 2 (5%) | 0 / 0 | 206 / 224 | 248 / 259 | 569 / 574 | 2 |
| prio (p95 390) | done | 32 (80%) | 0 / 0 | 0 / 0 | 18 / 122 | 115 / 252 | 0 |
| prio (p95 390) | running | 2 (5%) | 0 / 0 | 52 / 60 | 51 / 59 | 171 / 180 | 0 |
| prio (p95 390) | queued | 6 (15%) | 6 / 0 | 257 / 394 | 256 / 393 | 377 / 518 | 2 |
| cancel (p95 475) | done | 38 (95%) | 0 / 0 | 0 / 0 | 10 / 376 | 126 / 508 | 2 |
| cancel (p95 475) | queued | 2 (5%) | 0 / 2 | 0 / 1 | 339 / 340 | 473 / 474 | 0 |
| both (p95 266) | done | 32 (80%) | 0 / 0 | 0 / 1 | 2 / 115 | 113 / 238 | 0 |
| both (p95 266) | running | 4 (10%) | 0 / 0 | 38 / 72 | 38 / 71 | 165 / 202 | 0 |
| both (p95 266) | queued | 4 (10%) | 0 / 4 | 0 / 0 | 160 / 197 | 275 / 317 | 2 |
