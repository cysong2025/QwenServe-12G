# E09 Deadline-aware Admission Comparison

Generated at: 2026-08-30T06:50:07.233288+00:00

Overall frozen gate: **PASS**

Each gated holdout burst requires three valid exact-trace repetitions per policy, at least 15% measured offered headroom, deadline-aware P95 TTFT/TPOT within 1000/50 ms in every repetition, median goodput at least 10% above the better of unbounded and fixed-C4, Jain SLO-yield fairness at least 0.80, and no more than 0.05 fairness regression versus fixed-C4.

| Profile | Evidence | Offered/headroom | Goodput unbounded/fixed/deadline | Gain | P95 TTFT unbounded/fixed/deadline ms | Queue P95 ms | Fairness fixed/deadline | SLO | Fairness | Gate |
|---|---|---:|---:|---:|---:|---:|---:|---|---|---|
| holdout_nominal | VALID | 1.006/0.00% | 1.006/0.861/1.006 | 0.00% | 239.2/238.8/240.5 | 0.2 | 0.999/1.000 | PASS | PASS | NOT_APPLICABLE |
| holdout_periodic_burst | VALID | 2.072/45.70% | 1.422/1.050/1.689 | 18.75% | 2440.9/240.3/723.4 | 639.7 | 0.999/0.997 | PASS | PASS | PASS |
| holdout_shock_burst | VALID | 2.094/56.43% | 1.339/0.994/1.572 | 17.43% | 3120.4/240.4/739.2 | 660.7 | 0.996/0.998 | PASS | PASS | PASS |
