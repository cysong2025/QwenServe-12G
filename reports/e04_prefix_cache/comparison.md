# E04 Automatic Prefix Caching Comparison

Generated at: 2026-08-21T01:06:19.037414+00:00

OFF and ON use paired seeds. BENEFIT requires valid evidence, at least 5% lower P95 TTFT, no more than 2% output-throughput regression, and identical generated outputs.

| Condition | C | Prefix/Suffix | Nominal reuse | Actual token hit rate | OFF/ON P95 TTFT ms | TTFT delta | OFF/ON output tok/s | Throughput delta | Output | Evidence | SLO | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| capacity_reuse90_p1792 | 8 | 1792/256 | 90% | 78.93 [78.91, 78.94] | 1407.15/631.67 | -55.11% | 412.10/554.18 | +34.48% | MISMATCH | INCOMPLETE | UNKNOWN | UNKNOWN |
| reuse0_p1024 | 4 | 1024/1024 | 0% | 0.98 [0.98, 0.98] | 806.43/802.99 | -0.43% | 260.64/261.16 | +0.20% | MISMATCH | INCOMPLETE | UNKNOWN | UNKNOWN |
| reuse50_p1024 | 4 | 1024/1024 | 50% | 18.78 [18.30, 20.05] | 804.27/800.84 | -0.43% | 260.62/270.63 | +3.84% | MISMATCH | INCOMPLETE | UNKNOWN | UNKNOWN |
| reuse90_p1024 | 4 | 1024/1024 | 90% | 45.52 [45.52, 45.52] | 804.86/610.09 | -24.20% | 260.60/286.54 | +9.95% | MISMATCH | INCOMPLETE | UNKNOWN | UNKNOWN |
| reuse90_p1792 | 4 | 1792/256 | 90% | 78.94 [78.92, 78.95] | 804.81/436.33 | -45.79% | 260.66/309.23 | +18.63% | MISMATCH | INCOMPLETE | UNKNOWN | UNKNOWN |
| reuse90_p256 | 4 | 256/1792 | 90% | 12.11 [12.11, 12.11] | 807.90/734.78 | -9.05% | 260.57/266.88 | +2.42% | MISMATCH | INCOMPLETE | UNKNOWN | UNKNOWN |

## Data-dependent conclusion

No validated C4/P1024 reuse threshold currently satisfies the predefined benefit rule.
The C4/90% prefix-length sweep is not yet complete.
The C8 capacity validation is not yet complete.

Actual hit rate is calculated from the per-run delta of vLLM prefix-cache hit/query token counters. Peak VRAM is the maximum sampled value across three repetitions.
