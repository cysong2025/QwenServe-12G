# E07 Base vs LoRA Quality

Generated at: 2026-08-28T00:29:26.100068+00:00

Automated status: **PASS**
Human review: **PASS**
Overall status: **PASS**

Dataset SHA-256 match: YES
Prompt matches: 50/50

| State | Schema pass | Root cause Macro-F1 | Action micro-F1 | Dangerous command rate |
|---|---:|---:|---:|---:|
| Base | 92.00% | 0.7191 | 0.5208 | 0.00% |
| LoRA | 100.00% | 0.9366 | 0.9400 | 0.00% |

Frozen automated gate: LoRA schema >= 98%, root Macro-F1 >= 0.90, action micro-F1 >= 0.85, dangerous command rate = 0%; versus Base, schema may not regress, root Macro-F1 must improve >= 0.10, and action micro-F1 must improve >= 0.20.

## Blinded human review

Base mean score: 3.580
LoRA mean score: 4.760
LoRA - Base: +1.180

Frozen human gate: LoRA mean score must exceed Base by at least 0.30.
