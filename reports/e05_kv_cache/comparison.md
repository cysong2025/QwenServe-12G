# E05 FP8 KV Cache Performance Comparison

Generated at: 2026-08-21T18:27:15.863447+00:00

BF16 and FP8 use paired seeds and identical workload controls. Quality is evaluated separately and is not inferred from this table.

| Workload | In/Out | C | BF16 tok/s | FP8 tok/s | Delta | BF16 P95 TTFT | FP8 P95 TTFT | Delta | BF16 P95 TPOT | FP8 P95 TPOT | Delta | BF16 goodput | FP8 goodput | Delta | BF16/FP8 VRAM MiB | Exact output pairs | Evidence | FP8 SLO | Signal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| long | 2048/256 | 8 | 413.14 | 423.80 | +2.58% | 1630.31 | 1452.69 | -10.89% | 18.07 | 17.59 | -2.69% | 0.65 | 0.46 | -28.19% | 11723/11445 | 0/3 | VALID | FAIL | CONTROL |
| long | 2048/256 | 16 | 575.39 | 595.61 | +3.51% | 2867.29 | 2891.43 | +0.84% | 25.55 | 24.61 | -3.67% | 0.45 | 0.58 | +29.39% | 11714/11447 | 0/3 | VALID | FAIL | CONTROL |
| nearmax | 7168/256 | 8 | 185.11 | 197.16 | +6.51% | 4058.03 | 3963.57 | -2.33% | 39.29 | 36.71 | -6.58% | 0.09 | 0.10 | +6.50% | 11737/11447 | 0/3 | VALID | FAIL | NO_BENEFIT |
| nearmax | 7168/256 | 16 | 198.86 | 228.23 | +14.77% | 10675.42 | 9669.63 | -9.42% | 63.43 | 65.27 | +2.90% | 0.00 | 0.01 | NA | 11507/11449 | 0/3 | VALID | FAIL | BENEFIT |
| xlong | 4096/256 | 8 | 286.09 | 299.03 | +4.52% | 3133.54 | 3108.04 | -0.81% | 25.74 | 24.51 | -4.79% | 0.26 | 0.23 | -9.11% | 11714/11447 | 0/3 | VALID | FAIL | NO_BENEFIT |
| xlong | 4096/256 | 16 | 357.12 | 376.00 | +5.28% | 5813.95 | 5756.51 | -0.99% | 41.57 | 37.89 | -8.85% | 0.17 | 0.18 | +5.28% | 11714/11447 | 0/3 | VALID | FAIL | NO_BENEFIT |
