# E07 Base vs LoRA Online Cost

Generated at: 2026-08-28T00:29:26.036360+00:00

Overall online-cost gate: **FAIL**

Each cell requires three valid paired repetitions. The frozen cost gate requires LoRA SLO PASS, output throughput loss <= 20%, P95 TTFT increase <= 25%, and P95 TPOT increase <= 20%.

| Workload | In/Out | C | Base/LoRA tok/s | Delta | Base/LoRA P95 TTFT | Delta | Base/LoRA P95 TPOT | Delta | Base/LoRA goodput | Base/LoRA VRAM MiB | Evidence | LoRA SLO | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| medium | 512/256 | 1 | 86.75/72.42 | -16.53% | 58.94/74.87 | +27.02% | 11.43/13.66 | +19.50% | 0.34/0.28 | 10500/11048 | VALID | PASS | FAIL |
| medium | 512/256 | 4 | 311.24/271.46 | -12.78% | 209.85/249.04 | +18.68% | 12.67/14.40 | +13.64% | 1.22/1.06 | 10468/11031 | VALID | PASS | PASS |
| medium | 512/256 | 8 | 554.37/484.35 | -12.63% | 404.03/453.60 | +12.27% | 13.68/15.55 | +13.64% | 2.17/1.89 | 10597/11134 | VALID | PASS | PASS |
| short | 128/128 | 1 | 88.22/73.38 | -16.83% | 37.54/58.04 | +54.61% | 11.35/13.51 | +19.04% | 0.69/0.57 | 10472/11035 | VALID | PASS | FAIL |
| short | 128/128 | 4 | 322.93/279.51 | -13.45% | 72.14/102.99 | +42.75% | 12.16/13.87 | +14.03% | 2.52/2.18 | 10472/11035 | VALID | PASS | FAIL |
| short | 128/128 | 8 | 599.71/518.54 | -13.54% | 120.19/150.92 | +25.56% | 12.59/14.41 | +14.43% | 4.69/4.05 | 10500/11035 | VALID | PASS | FAIL |
