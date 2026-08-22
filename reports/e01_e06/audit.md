# E01-E06 Evidence Audit

Generated at: 2026-08-22T11:59:37.208896+00:00

Overall status: **PASS**
Milestone status: **E01_E06_COMPLETE**
Unique formal benchmark runs: **228**

A PASS means the committed evidence is complete and internally consistent. Expected negative findings, including the E05 quality regression, remain failures of their optimization gates and are not rewritten as successful optimizations.

| Experiment | Status | Unique runs | Profiles | Checks |
|---|---|---:|---:|---:|
| E01 | COMPLETE_WITH_PROTOCOL_DEVIATION | 36 | 12 | 9/9 |
| E02 | COMPLETE | 72 | 24 | 10/10 |
| E03 | COVERED_BY_E01 | 0 | 0 | 3/3 |
| E04 | COMPLETE_WITH_LIMITATIONS | 36 | 12 | 13/13 |
| E05 | COMPLETE_WITH_QUALITY_REGRESSION | 36 | 12 | 12/12 |
| E06 | COMPLETE | 48 | 16 | 14/14 |

## Frozen Findings

- E01/E03: 8 of 12 baseline profiles pass SLO; C16 and Long-C8 expose the queueing boundary.
  The planned Transformers single-request reference was not executed, so no vLLM-versus-Transformers speedup is claimed.
- E02: all four Long-C8 budgets fail the every-repetition SLO gate; smaller budgets reduce long-input TTFT but do not replace admission control.
- E04: APC functional equivalence passes 24/24 canary cases, while the strict random-output gate remains incomplete and task accuracy remains 22/24 on both sides.
- E05: FP8 KV capacity is 2.009x, but automated and blinded-human quality gates fail.
- E06: stacked benefit is validated for reuse50_p1024/C4 and capacity_reuse90_p1792/C8; the four-cell canary passes 24/24.
