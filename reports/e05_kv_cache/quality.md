# E05 FP8 KV Cache Quality

Generated at: 2026-08-22T04:18:36.006894+00:00
Finalized at: 2026-08-22T04:18:36.058132+00:00

Automated status: **FAIL**
Human review: **FAIL**
Overall status: **FAIL**

Dataset SHA-256 match: YES
Prompt matches: 50/50
Raw BF16/FP8 output matches: 2/50

| State | Schema pass | Root cause Macro-F1 | Action micro-F1 | Dangerous command rate |
|---|---:|---:|---:|---:|
| BF16 | 92.00% | 0.7191 | 0.5208 | 0.00% |
| FP8 | 70.00% | 0.5698 | 0.3647 | 0.00% |

Frozen automated gate: BF16 schema >= 90%, root Macro-F1 >= 0.80, action micro-F1 >= 0.75, dangerous commands <= 2%; FP8 may drop at most 0.02 on each quality score and may not exceed a 2% dangerous-command rate.

## Blinded human review

BF16 mean score: 3.680
FP8 mean score: 3.120
FP8 - BF16: -0.560
Preferences: BF16 20, FP8 5, tie 25

Frozen human gate: FP8 mean score must be no more than 0.10 below BF16.
