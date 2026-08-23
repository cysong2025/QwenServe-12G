# Codex Agent Handoff

Last updated: 2026-08-24

This document is the handoff entry point for continuing QwenServe-12G in a new
Codex task. Read it before changing code or asking the user to run GPU work.
The repository and the latest user-provided output remain the source of truth
when they differ from this snapshot.

## 1. Project objective

QwenServe-12G is a reproducible AI infrastructure study for serving and
parameter-efficient training on one RTX 5070 12 GB GPU under WSL2. It is not a
model-serving demo. Every optimization must have a frozen question, controlled
comparison, repeated measurements, quality checks, telemetry, and an explicit
deployment boundary.

The main question is:

> Under RTX 5070 12 GB, WSL2, and Qwen2.5-3B-Instruct, how can single-GPU SLO
> goodput be improved without violating latency or model-quality constraints?

## 2. Current handoff state

- GitHub repository: `git@github.com:cysong2025/QwenServe-12G.git`
- Active branch: `codex/e07-lora`
- Last prepared commit: `e4ad048 Prepare reproducible E07 QLoRA and LoRA experiments`
- Mac workspace: `/Users/songchuangye/Documents/推理训练`
- WSL2 workspace: `~/projects/QwenServe-12G`
- E01-E06: complete; 228 formal benchmark runs; structured audit `PASS`.
- E07: code, protocol, generated data, tests, and runbook are ready; status is
  `READY_FOR_GPU`; GPU training and serving results have not been run.
- E08 token-aware admission control: future milestone, not started.
- CPU/static verification at handoff: 70 tests passed.
- Planned E07 online matrix: 36 formal runs, comprising Base and rank-8 LoRA,
  six workload/concurrency cells each, three repetitions per cell.

Do not describe E07 as complete and do not claim LoRA quality gains or online
cost until `reports/e07_lora/final.md` has been generated from GPU evidence.

One user-owned untracked file existed at handoff:

```text
reports/e05_kv_cache/human_review.backup.csv
```

Do not edit, stage, delete, or rename it unless the user explicitly requests
that action.

## 3. Completed evidence and honest boundaries

| Stage | Topic | Current conclusion |
|---|---|---|
| E01/E03 | Baseline and length x concurrency | 8 of 12 profiles pass SLO; long-context high concurrency raises throughput but can reduce goodput. |
| E02 | Batch token budget | Budget 2048 substantially lowers Long-C4/C8 TTFT relative to 8192 with similar throughput. |
| E04 | Automatic Prefix Caching | APC improves TTFT when real prefix reuse is present; retain the documented random-token correctness limitation. |
| E05 | FP8 KV cache | KV capacity is about 2.009x, but schema quality fell from 92% to 70% and blind-review score from 3.680 to 3.120; FP8 is not the default recommendation. |
| E06 | Combined optimization | Benefits are workload-dependent; frozen correctness canary passed 24/24. |
| E07 | QLoRA and LoRA serving | Preparation only; no measured result yet. |

Important truth boundaries:

- E01 did not run the planned Transformers single-request reference. Do not
  claim a cross-engine vLLM speedup.
- A scientific gate failure is a valid negative result. Do not loosen frozen
  thresholds, delete evidence, or relabel a failed gate as success.
- `make compare-*` may exit with code 2 when an experiment fails its scientific
  gate. This does not by itself mean the comparison code is broken.
- Preserve E04 and E05 limitations in reports, resumes, and interview answers.

Primary completed references:

- `docs/E01_E06_FINAL_REPORT.md`
- `docs/INTERVIEW_GUIDE_E01_E06.md`
- `reports/final/e01_e06_audit.md`

## 4. Target environment

| Component | Frozen/observed value |
|---|---|
| Host | Windows with Ubuntu 24.04 WSL2 |
| GPU | NVIDIA GeForce RTX 5070, 12227 MiB |
| GPU architecture | Blackwell `sm_120` |
| Windows driver | 581.29 |
| CUDA reported by driver | 13.0 |
| RAM / swap | 24 GiB / 16 GiB |
| Python | 3.12 |
| vLLM | 0.25.1 |
| PyTorch | 2.11.0+cu130 |
| Model | Qwen/Qwen2.5-3B-Instruct |
| Local model path | `~/models/Qwen2.5-3B-Instruct` |

E07 training dependencies are deliberately pinned:

- `accelerate==1.12.0`
- `bitsandbytes==0.49.1`
- `peft==0.18.1`

The setup script installs them with `--no-deps` to avoid silently replacing
the working PyTorch, Transformers, or CUDA stack, then runs `pip check`.

## 5. Known environment and network issues

### Git and model downloads

- GitHub HTTPS has repeatedly timed out from WSL2. Git over SSH works and must
  remain the default. Do not switch the remote back to HTTPS.
- `huggingface.co:443` is blocked or reset from both Windows and WSL2 on the
  current network. Port 22 cannot replace HTTPS for Hugging Face model APIs.
- Use the existing local ModelScope model snapshot. Avoid code paths that try
  to fetch model configuration, tokenizer, or weights from Hugging Face.
- WSL2 mirrored networking and DNS tunneling were enabled, but they did not
  make Hugging Face reachable.

### RTX 5070 and FlashInfer

The RTX 5070 hardware supports `sm_120`. The earlier startup failure was a
FlashInfer sampler compatibility/architecture-recognition issue in this
software stack, not a GPU that lacks modern compute capability.

All controlled serving configs set:

```text
VLLM_USE_FLASHINFER_SAMPLER=0
```

This only replaces the FlashInfer sampling path with the PyTorch sampler; it
does not disable every FlashInfer feature. The attention backend is pinned to
`TRITON_ATTN`. Do not remove these controls while comparing profiles.

### Other established fixes

- PyArrow is pinned to `20.0.0` because the installed `datasets` code expects
  `PyExtensionType`.
- E06 used `gpu_memory_utilization = 0.78` after an OOM investigation.
- `WARNING: /tokenize unavailable, skipping alignment.` is usually non-fatal
  for the random benchmark. A later connection refusal means the server died;
  inspect the first server-side exception instead of blaming this warning.
- `Engine core initialization failed` is a wrapper symptom. Diagnose the
  earliest root cause in terminal 1 or the server log, especially OOM,
  FlashInfer, local-path, or dependency errors.

## 6. Operating model with the user

The user develops through Codex on Mac and manually executes GPU commands on
Windows/WSL2, often over SSH. The new Agent should:

1. Make and test CPU/static code changes on Mac.
2. Commit and push them to `codex/e07-lora` over Git SSH.
3. Give direct, copyable commands in execution order, not only a manual link.
4. Clearly label terminal 1 as the long-running server and terminal 2 as the
   benchmark/evaluation terminal.
5. Ask the user to paste the earliest relevant error or generated report, then
   diagnose that evidence before changing frozen settings.
6. Never imply that Mac executed RTX 5070 GPU work.

Do not run training and vLLM serving at the same time. Before training:

```bash
pgrep -af 'vllm serve' || true
nvidia-smi
```

Only one server profile may listen on port 8000. Stop terminal 1 with `Ctrl+C`
before switching Base/LoRA profiles or starting training. A healthy server is
ready only after `Application startup complete` and a successful `/v1/models`
request.

## 7. E07 design and frozen gates

E07 asks whether QLoRA training can improve a frozen 50-case AI-infrastructure
incident triage task, and what serving cost the resulting LoRA Adapter adds
relative to the same BF16 Base model.

### Training roles

- Rank 8 is the primary Adapter used in quality and online performance tests.
- Rank 16 is a predeclared training ablation.
- Do not choose rank 16 as the primary Adapter after looking at outcomes.
- Adapter directories under `artifacts/adapters/e07/` are ignored by Git.
- Training scripts preserve failed-attempt manifests and `nvidia-smi`
  telemetry.
- The trainer refuses to overwrite a non-empty Adapter directory. Archive an
  old attempt with `mv`; do not automatically delete it.
- Serving validates Adapter rank plus the manifest-authenticated weights hash
  before startup.

### Quality gates

- JSON/schema compliance: at least 98%.
- Root-cause Macro-F1: at least 0.90.
- Action micro-F1: at least 0.85.
- Dangerous-command count: 0.
- Versus Base: no schema regression, root-cause Macro-F1 improvement at least
  0.10 absolute, action micro-F1 improvement at least 0.20 absolute.
- Blinded human review: LoRA mean at least Base mean plus 0.30.

Do not read `human_review_key.json` before all 50 blind-review rows are scored.
The user should ideally perform this review. If the user explicitly delegates
it to the Agent, preserve blindness and record that limitation.

### Online serving gates

Every one of the six LoRA matrix cells must:

- pass the frozen TTFT/TPOT SLO;
- lose no more than 20% output throughput versus Base;
- increase P95 TTFT by no more than 25%;
- increase P95 TPOT by no more than 20%;
- keep error rate below 1%.

The cells are short `128/128` and medium `512/256`, each at concurrency 1, 4,
and 8, with three repetitions. The pilot cell belongs to the matrix and valid
pilot repetitions are skipped by the formal matrix runner.

## 8. Exact continuation sequence

### Step A: update and verify WSL2 checkout

```bash
cd ~/projects/QwenServe-12G
git remote set-url origin git@github.com:cysong2025/QwenServe-12G.git
git fetch origin
git switch codex/e07-lora
git pull --ff-only origin codex/e07-lora

source .venv/bin/activate
make test
make prepare-e07-data
make audit-e07-readiness
sed -n '1,220p' reports/e07_lora/readiness.md
```

Continue only if readiness is `READY_FOR_GPU`.

### Step B: install pinned training dependencies

```bash
make install-e07-train-deps
python -c 'import accelerate, bitsandbytes, peft; print(accelerate.__version__, bitsandbytes.__version__, peft.__version__)'
```

Do not upgrade unrelated packages between Base and LoRA measurements.

### Step C: run QLoRA smoke

```bash
pgrep -af 'vllm serve' || true
nvidia-smi
make render-e07-smoke
make train-e07-smoke

PYTHONPATH=src python3 -m qwen_serve_lab.cli inspect-e07-adapter \
  --adapter-dir artifacts/adapters/e07/smoke-r8 \
  --expected-rank 8 \
  --output-dir reports/e07_lora/smoke
```

Then validate loading. In terminal 1:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e07-lora-local E07_ADAPTER_PATH=artifacts/adapters/e07/smoke-r8
```

After startup, in terminal 2:

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

The model list must contain `ai-infra-triage-r8`. Stop terminal 1 before formal
training.

### Step D: formal rank-8 and rank-16 training

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
pgrep -af 'vllm serve' || true
nvidia-smi

make train-e07-rank8
make inspect-e07-adapter
sed -n '1,160p' reports/e07_lora/adapter.md

make train-e07-rank16
PYTHONPATH=src python3 -m qwen_serve_lab.cli inspect-e07-adapter \
  --adapter-dir artifacts/adapters/e07/rank16 \
  --expected-rank 16 \
  --output-dir reports/e07_lora/rank16
```

### Step E: Base measurements

Terminal 1:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e07-base-local
```

Terminal 2, only after startup completes:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make bench-e07-base-pilot
make run-e07-quality-base
make bench-e07-base-matrix
```

Stop terminal 1 after all three commands finish.

### Step F: rank-8 LoRA measurements

Terminal 1:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e07-lora-local
```

Terminal 2, only after startup completes:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make bench-e07-lora-pilot
make run-e07-quality-lora
make bench-e07-lora-matrix
```

Stop terminal 1 when complete.

### Step G: comparisons, blind review, and final report

```bash
make summarize-e07
make compare-e07
make compare-e07-quality
sed -n '1,220p' reports/e07_lora/comparison.md
sed -n '1,220p' reports/e07_lora/quality.md
```

Complete all blank scoring fields in
`reports/e07_lora/human_review.csv` without opening the blind key. Then:

```bash
make summarize-e07-human-review
make finalize-e07
sed -n '1,220p' reports/e07_lora/final.md
```

Detailed operator commands and failure handling live in
`docs/M3_E07_QLORA_LORA_RUNBOOK.md`.

## 9. Expected evidence and completion criteria

Expected local-only outputs include:

- `artifacts/adapters/e07/smoke-r8/`
- `artifacts/adapters/e07/rank8/`
- `artifacts/adapters/e07/rank16/`
- `artifacts/results/e07_training/`
- detailed Base and LoRA benchmark/quality JSON

Expected compact report outputs include:

- `reports/e07_lora/adapter.json` and `adapter.md`
- `reports/e07_lora/runs.csv` and `summary.md`
- `reports/e07_lora/comparison.csv` and `comparison.md`
- `reports/e07_lora/quality.json` and `quality.md`
- `reports/e07_lora/human_review.csv`
- `reports/e07_lora/human_review_summary.json` and `.md`
- `reports/e07_lora/final.json` and `final.md`

E07 is complete only when:

1. Smoke, rank-8, and rank-16 training evidence is preserved.
2. The primary rank-8 Adapter passes structural/hash inspection.
3. Base and LoRA quality evaluations exist for the frozen dataset.
4. All 36 planned performance repetitions are accounted for and valid.
5. Blind review is complete without unblinding during scoring.
6. `final.md` records pass or fail against every frozen gate.
7. Compact reproducible evidence is committed; negative results remain intact.

Do not commit model weights, Adapter `.safetensors`, or detailed request-level
artifacts. Raw artifacts stay on the WSL2 host. Commit only the compact reports
and manifests listed by the E07 runbook.

## 10. Important code and document map

- `docs/E07_PROTOCOL.md`: research question, frozen protocol, and gates.
- `docs/E07_DATA_CARD.md`: generated training/validation/test data provenance.
- `docs/M3_E07_QLORA_LORA_RUNBOOK.md`: authoritative manual GPU sequence.
- `docs/E07_RESULTS_TEMPLATE.md`: report shape only, not measured evidence.
- `reports/e07_lora/readiness.md`: pre-GPU audit status.
- `configs/train/e07_*.toml`: smoke, rank-8, and rank-16 training configs.
- `configs/serve/e07_base.toml`: Base serving profile.
- `configs/serve/e07_lora.toml`: primary rank-8 LoRA serving profile.
- `configs/matrix/e07_*.toml`: frozen Base/LoRA performance matrices.
- `src/qwen_serve_lab/e07*.py`: data, training, inspection, quality, comparison,
  blind-review, readiness, and finalization logic.
- `Makefile`: operator-facing entry points.

Before editing, inspect `git status`, the latest commit, relevant tests, and the
latest user output. Work with existing user changes and never reset or clean a
dirty worktree automatically.

## 11. New-Agent startup prompt

The user can start the next Codex task with:

```text
继续 QwenServe-12G 项目。请先阅读 docs/CODEX_AGENT_HANDOFF.md，检查当前
git status、分支和最新提交，再阅读 E07 runbook/readiness。E01-E06 已完成，
当前目标是协助我在 Windows/WSL2 手工完成 E07 GPU 实验；Mac 端负责代码、
测试、报告和 Git 提交。请直接按步骤给可复制命令，分析我粘贴的输出，保持
冻结实验门槛，不要触碰 reports/e05_kv_cache/human_review.backup.csv，也不要
把尚未运行的 E07 描述为已完成。
```
