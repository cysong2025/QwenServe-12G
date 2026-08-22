# E06 Combined Optimization Factorial Comparison

Generated at: 2026-08-22T11:45:28.490461+00:00

Cells are A=8192/OFF, B=2048/OFF, C=8192/ON, and D=2048/ON. STACKED_BENEFIT requires valid four-cell evidence, D throughput no more than 2% below the better single treatment, and either at least 5% lower P95 TTFT or at least 10% higher goodput than the better single treatment.
Random-token output hashes are diagnostic only; fixed canary equivalence is reported separately.

| Condition | C | Reuse | APC hit C/D | P95 TTFT A/B/C/D ms | D vs best | Output tok/s A/B/C/D | D vs best | Goodput D vs best | Output matches B/C/D vs A | Evidence | D SLO | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|
| capacity_reuse90_p1792 | 8 | 90% | 78.94/78.94 | 1513.23/845.64/670.23/597.97 | -10.78% | 447.71/447.54/618.11/618.77 | +0.11% | +0.11% | 0/3/0/3/0/3 | VALID | PASS | STACKED_BENEFIT |
| reuse0_p1024 | 4 | 0% | 0.98/0.98 | 776.99/628.45/761.28/630.51 | +0.33% | 287.75/287.15/288.87/288.11 | -0.26% | -0.26% | 0/3/0/3/0/3 | VALID | PASS | NO_STACKED_BENEFIT |
| reuse50_p1024 | 4 | 50% | 19.27/20.26 | 768.98/755.28/755.21/577.36 | -23.55% | 288.20/287.00/298.54/299.39 | +0.28% | +0.28% | 0/3/0/3/0/3 | VALID | PASS | STACKED_BENEFIT |
| reuse90_p1024 | 4 | 90% | 45.54/45.54 | 767.03/581.46/485.15/466.82 | -3.78% | 288.67/289.27/319.67/317.67 | -0.62% | -0.62% | 0/3/0/3/0/3 | VALID | PASS | NO_STACKED_BENEFIT |

## Factorial interaction

Interaction is the APC percentage effect at budget 2048 minus the APC percentage effect at budget 8192. Negative TTFT interaction and positive throughput interaction indicate that APC becomes more effective with the smaller scheduler budget.

| Condition | APC TTFT effect 8192 | APC TTFT effect 2048 | TTFT interaction | APC throughput effect 8192 | APC throughput effect 2048 | Throughput interaction |
|---|---:|---:|---:|---:|---:|---:|
| capacity_reuse90_p1792 | -55.71% | -29.29% | 26.42 pp | +38.06% | +38.26% | 0.20 pp |
| reuse0_p1024 | -2.02% | +0.33% | 2.35 pp | +0.39% | +0.33% | -0.06 pp |
| reuse50_p1024 | -1.79% | -23.56% | -21.77 pp | +3.59% | +4.32% | 0.73 pp |
| reuse90_p1024 | -36.75% | -19.72% | 17.03 pp | +10.74% | +9.82% | -0.92 pp |

## Data-dependent conclusion

Validated stacked benefit is present at: capacity_reuse90_p1792/c8, reuse50_p1024/c4.
For the no-reuse control, combined D versus budget-only B changes P95 TTFT by +0.33% and output throughput by +0.33%.
The C8/P1792 capacity condition is STACKED_BENEFIT with combined SLO PASS.
