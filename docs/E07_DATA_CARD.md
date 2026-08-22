# E07 Data Card

## Purpose

The dataset trains and evaluates an AI infrastructure incident triage model that
returns a strict JSON object with one root-cause label, exactly two action labels,
and a dangerous-command flag. It is designed for controlled systems experiments,
not as a production incident corpus.

## Composition

| Split | Rows | Source groups | Use |
|---|---:|---:|---|
| Train | 250 | 50 | QLoRA optimization |
| Validation | 100 | 20 | Loss monitoring and model selection evidence |
| Test | 50 | 50 frozen cases | Base/LoRA quality comparison only |

There are ten balanced root-cause classes. Each training/validation source
scenario is rendered through five deterministic operational-context prefixes.
Train and validation use disjoint scenario groups. The frozen test set is the
same independently written set used by E05 and is never emitted into SFT files.

## Provenance

All examples are synthetic and authored for this repository. They describe
common failure modes such as CUDA OOM, KV cache exhaustion, dependency conflict,
artifact unavailability, overload, and unsupported precision. They do not contain
customer incidents, personal data, credentials, or proprietary logs.

## Generation and integrity

Run:

```bash
make prepare-e07-data
```

The generated `datasets/e07_sft/dataset_manifest.json` records source, train,
validation, and frozen-test SHA-256 values. Generation fails on duplicate source
incidents, label mismatch, class imbalance, split-group overlap, or exact overlap
with a frozen test incident.

## Limitations

- Synthetic wording is cleaner than production telemetry and may overestimate
  generalization.
- Five render views share one underlying scenario, so row count is not the same
  as independent semantic diversity.
- Labels force one root cause and two actions even when real incidents can be
  multi-causal.
- The 50-case test set supports controlled comparison, not broad safety claims.
- Test results must not be used to rewrite source scenarios during the formal E07
  run. A new data revision requires a new protocol version and complete rerun.
