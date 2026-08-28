# E08 Admission Strategy Comparison

Generated at: 2026-08-28T17:01:21.943559+00:00

Overall frozen gate: **FAIL**

The two overload profiles require three valid exact-trace repetitions per policy, token-aware P95 TTFT/TPOT within 1000/50 ms in every repetition, median goodput at least 10% above the better of unbounded and fixed-C4, Jain SLO-yield fairness at least 0.80, and no more than 0.05 fairness regression versus fixed-C4.

| Profile | Evidence | Goodput unbounded/fixed/token | Token gain | P95 TTFT unbounded/fixed/token ms | Fairness unbounded/fixed/token | Token SLO | Token fairness | Gate |
|---|---|---:|---:|---:|---:|---|---|---|
| mixed_burst | VALID | 1.500/0.994/1.339 | -10.74% | 2597.8/242.5/243.2 | 0.999/0.994/0.972 | PASS | PASS | FAIL |
| mixed_nominal | VALID | 0.956/0.850/0.889 | -6.98% | 251.1/250.1/248.4 | 1.000/1.000/0.994 | PASS | PASS | NOT_APPLICABLE |
| mixed_overload | VALID | 1.811/1.189/1.533 | -15.34% | 900.5/248.3/244.2 | 1.000/0.997/0.957 | FAIL | PASS | FAIL |
