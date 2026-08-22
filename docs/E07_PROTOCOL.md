# E07 QLoRA and LoRA Serving Protocol

## Background

E01-E06 established the capacity, scheduling, cache, and precision behavior of
Qwen2.5-3B-Instruct on one RTX 5070. Those experiments improve serving behavior
but do not improve task capability. E07 closes that gap with parameter-efficient
training for AI infrastructure incident triage and then measures the Adapter's
online cost under the same serving controls.

## Research question

Can a rank-8 QLoRA Adapter materially improve fixed incident-triage quality on a
12 GB consumer GPU while preserving the latency SLO and keeping online overhead
within a frozen operational budget?

The hypothesis is not a conclusion. E07 fails if quality does not improve, if
the Adapter cannot be loaded reproducibly, or if any formal serving cell exceeds
the online-cost gate.

## Controlled treatment

- Base model: `Qwen/Qwen2.5-3B-Instruct` at revision
  `a1d308dfcc03e09da285d49d912439a655a571e8`.
- Primary treatment: 4-bit NF4 QLoRA, BF16 compute, rank 8, alpha 16.
- Secondary training ablation: rank 16, alpha 32. It is not substituted into
  the primary performance matrix after seeing results.
- Target modules: attention projections and MLP projections.
- Sequence length 1024, micro batch 1, gradient accumulation 16, gradient
  checkpointing enabled.
- Training and vLLM serving run at different times.

## Data protocol

- `datasets/e07_ai_infra_sft_source.json` contains grouped synthetic training
  and validation scenarios.
- The deterministic builder emits 250 balanced train rows and 100 balanced
  validation rows from disjoint scenario groups.
- `datasets/e05_ai_infra_quality.json` remains the frozen 50-case test set.
- The builder rejects exact source-incident overlap with the test set and stores
  SHA-256 values for all three splits.
- Test results may not be used to edit training examples after a formal run.

## Training gates

1. The 100-example smoke produces `adapter_model.safetensors`,
   `adapter_config.json`, and `training_manifest.json`.
2. vLLM loads the smoke Adapter and exposes its model name.
3. The full rank-8 run completes without an active vLLM process and records
   config, dataset, model, environment, Git, metrics, and weight hashes.
4. Adapter inspection verifies rank and file integrity before serving.

## Automated quality gate

Both Base and LoRA run the same 50 prompts with temperature zero and paired
seeds. LoRA must satisfy every condition:

- schema pass rate at least 98%;
- root-cause Macro-F1 at least 0.90;
- action micro-F1 at least 0.85;
- dangerous-command rate exactly 0%;
- schema does not regress from Base;
- root-cause Macro-F1 improves by at least 0.10 absolute;
- action micro-F1 improves by at least 0.20 absolute.

## Human quality gate

The same 50 Base/LoRA output pairs are anonymized as A/B. Review scores are
integers from 1 to 5. LoRA mean score must exceed Base by at least 0.30. The
blind key is generated before scores are entered.

## Online performance gate

Base and LoRA use matched BF16 KV, APC OFF, batch-token budget 2048,
`gpu_memory_utilization=0.78`, and the same six workload cells:

| Workload | Input/output | Concurrency |
|---|---:|---:|
| short | 128/128 | 1, 4, 8 |
| medium | 512/256 | 1, 4, 8 |

Each state/cell has three repetitions and 100 timed requests, for 36 formal
runs. Every LoRA cell must pass the original TTFT/TPOT SLO and satisfy:

- output throughput loss no more than 20%;
- P95 TTFT increase no more than 25%;
- P95 TPOT increase no more than 20%;
- error rate below 1% in every repetition.

## Conclusion rule

E07 is successful only when Adapter integrity, automated quality, blinded-human
quality, and all six online-cost cells pass. A quality gain with unacceptable
serving overhead is reported as a tradeoff, not as a successful deployment.
