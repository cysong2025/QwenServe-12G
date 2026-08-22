# E07 Results Template

Status: **GPU EXECUTION PENDING**

This file defines the final narrative shape. It is not an experimental result.

## Training evidence

| Profile | Rank | Train rows | Steps/epochs | Final train loss | Validation loss | Peak VRAM | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| smoke | 8 | 100 | pending | pending | pending | pending | pending |
| primary | 8 | 250 | pending | pending | pending | pending | pending |
| ablation | 16 | 250 | pending | pending | pending | pending | pending |

## Quality evidence

Populate from `reports/e07_lora/quality.md` and the blinded review. Report
absolute scores and LoRA-minus-Base deltas for schema, root Macro-F1, action
micro-F1, dangerous-command rate, and mean human score.

## Online cost

Populate all six cells from `reports/e07_lora/comparison.md`. Include Base and
LoRA output throughput, P95 TTFT, P95 TPOT, goodput, peak VRAM, and percent deltas.

## Reproducible conclusion

Use this form only after `make finalize-e07`:

> On the pinned RTX 5070/vLLM environment and frozen 50-case incident-triage
> task, the rank-8 Adapter changed quality by [measured deltas]. Under [six
> workload cells], it changed online metrics by [measured ranges], with [gate
> status]. The conclusion applies to [measured boundary] and does not establish
> [unmeasured generalization].
