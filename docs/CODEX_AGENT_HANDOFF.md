# Codex Agent Handoff

Last updated: 2026-08-31

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
- Active branch: `codex/open-source-readme` (based on the completed
  `codex/e09-deadline-admission` result branch).
- Mac workspace: `/Users/songchuangye/Documents/推理训练`
- WSL2 E08 workspace: `~/projects/QwenServe-12G-e08`
- WSL2 E09 workspace: `~/projects/QwenServe-12G-e09`
- E01-E06: complete; 228 formal benchmark runs; structured audit `PASS`.
- E07: execution complete; 36 formal benchmark runs are `VALID`, automated and
  delegated-Agent blind quality gates pass, but four of six online-cost cells
  fail the frozen TTFT overhead limit. Final machine status is `FAIL`.
- E08: execution complete; 3 pilot and 27 formal runs are `VALID`, exact-trace
  pairing and fairness pass, but both overload goodput gates fail and one
  token-aware repetition exceeds the frozen TPOT SLO. Final status is `FAIL`.
- E09: execution complete; calibration and 27/27 formal cells are `VALID`, all
  trace groups pair exactly, and both gated burst profiles pass goodput,
  headroom, absolute SLO, error-rate, and fairness gates. Final status is
  `PASS`.
- CPU/static verification at handoff: 91 tests passed.
- The public README now presents the architecture, measured results, quick
  start, and limitations with four report-derived SVGs. `make check-charts`
  and GitHub Actions prevent chart/report drift.
- The obsolete E07 result template and redundant `reports/.gitkeep` were
  removed; compact scientific CSV/JSON/Markdown evidence remains committed.
- Completed E07 online matrix: 36 formal runs, comprising Base and rank-8 LoRA,
  six workload/concurrency cells each, three repetitions per cell; error rate 0.

E07 execution is complete, but do not relabel its overall `FAIL` as successful
deployment. The measured quality gain coexists with unacceptable online TTFT
overhead in four frozen cells.

E08 execution is also complete, but the tested token-aware policy is not a
successful default. Burst/steady-overload median goodput gains versus the best
baseline are -10.74%/-15.34%, and steady-overload repetition 2 has 63.37 ms
P95 TPOT versus the frozen 50 ms limit.

E09 supplies the first repeated, independent-holdout positive admission result.
Deadline-aware median SLO-goodput gains versus the best baseline are +18.75%
for periodic burst and +17.43% for shock burst. The claim is bounded to the
frozen bursty traffic, workload mix, SLO, and RTX 5070 server profile; nominal
traffic shows 0% gain and E08 steady overload remains negative.

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
| E07 | QLoRA and LoRA serving | Adapter/automated quality/delegated-Agent blind quality PASS; online cost FAIL in short C1/C4/C8 and medium C1, so current dynamic LoRA is not the default deployment. |
| E08 | Token-aware admission | 27/27 formal cells are valid and paired; fairness PASS, but both overload goodput gates and one per-repetition TPOT SLO fail, so the current policy is not the default deployment. |
| E09 | Deadline-aware admission | 27/27 formal cells are valid and exact-paired; periodic/shock burst median goodput improves +18.75%/+17.43% versus the best baseline, with absolute SLO and fairness PASS. |

Important truth boundaries:

- E01 did not run the planned Transformers single-request reference. Do not
  claim a cross-engine vLLM speedup.
- A scientific gate failure is a valid negative result. Do not loosen frozen
  thresholds, delete evidence, or relabel a failed gate as success.
- `make compare-*` may exit with code 2 when an experiment fails its scientific
  gate. This does not by itself mean the comparison code is broken.
- Preserve E04 and E05 limitations in reports, resumes, and interview answers.
- Preserve the E07 reviewer limitation: the user delegated the 50-pair blind
  scoring to Codex Agent. A/B mapping remained hidden until all scores were
  locked, but this is not an independent-human preference study.

Primary completed references:

- `docs/E01_E06_FINAL_REPORT.md`
- `docs/INTERVIEW_GUIDE_E01_E06.md`
- `reports/final/e01_e06_audit.md`
- `docs/E07_RESULTS.md`
- `reports/e07_lora/final.md`
- `docs/E09_RESULTS.md`
- `reports/e09_deadline_admission/comparison.md`
- `reports/e09_deadline_admission/final.json`

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

The user develops through Codex on Mac and has authorized direct work on the
Windows/WSL2 GPU lab through the `ai-fpga-windows` SSH alias. The new Agent
should:

1. Make and test CPU/static code changes on Mac.
2. Commit and push them to the active `codex/*` branch over Git SSH.
3. Use the Windows/WSL lab helper for non-interactive remote work when safe;
   still give direct, copyable commands when user action is needed.
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

## 8. E08 execution record and reproduction sequence

E08 target-GPU execution is complete on `codex/e08-admission`. The sequence
below is retained for reproduction; the authoritative protocol remains
`docs/M4_E08_ADMISSION_RUNBOOK.md`, and the measured interpretation is in
`docs/E08_RESULTS.md`.

First update and inspect the WSL checkout. When using the Windows/WSL SSH lab,
probe the connection and inspect remote branch/status/revision before any sync.
Do not pull across a dirty remote worktree.

```bash
cd ~/projects/QwenServe-12G
git fetch origin
git switch codex/e08-admission
git pull --ff-only origin codex/e08-admission
source .venv/bin/activate
make test
make prepare-e08-traces
make audit-e08-readiness
sed -n '1,220p' reports/e08_admission/readiness.md
```

Continue only if the audit says `READY_FOR_GPU`, while still stating
`GPU execution: DEFERRED` and `Scientific result: NOT_RUN`.

Terminal 1 starts the controlled E06 Base/BF16 server and remains running:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e08-local
```

After `Application startup complete`, Terminal 2 verifies the endpoint and
runs the three-policy pilot:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
curl -fsS http://127.0.0.1:8000/v1/models | python -m json.tool
make bench-e08-pilot-unbounded
sleep 30
make bench-e08-pilot-fixed
sleep 30
make bench-e08-pilot-token
```

Only after all three pilot documents are `valid: true` and share a trace hash,
run the 27-cell formal matrix:

```bash
make bench-e08-matrix
```

The matrix uses three 180-second profiles, three policies, three repetitions,
balanced policy order, and 30-second cooldowns. It refuses to overwrite an
existing cell and stops on invalid evidence. Resume preserved cells with:

```bash
make bench-e08-matrix-resume
```

After all formal runs:

```bash
make compare-e08; status=$?; test "$status" -eq 0 -o "$status" -eq 2
sed -n '1,240p' reports/e08_admission/comparison.md
```

Exit code 2 may be a valid negative scientific result. Do not tune frozen
rates, budgets, SLOs, fairness thresholds, or the 10% best-baseline goodput
gate after seeing formal data.

The completed comparison returned 2 with `reports/e08_admission/final.json`
status `FAIL`. This is the expected machine representation of the preserved
negative scientific result, not an execution failure.

## 9. E08 expected evidence and completion criteria

Expected WSL-only raw outputs include:

- `artifacts/results/e08_admission/traces/` deterministic request traces;
- `artifacts/results/e08_admission/runs/` request-level policy decisions,
  streaming timings, controller state, environment snapshots, and telemetry.

Expected compact report outputs include:

- `reports/e08_admission/runs.csv`;
- `reports/e08_admission/comparison.csv` and `comparison.md`;
- `reports/e08_admission/final.json`;
- `reports/e08_admission/readiness.json` and `readiness.md`.

E08 is complete only when all 27 formal cells exist exactly once, all policies
share the same trace hash per profile/repetition, evidence validity passes,
and the frozen report records either `PASS` or a preserved scientific `FAIL`.
Readiness and pilot output alone are never E08 completion evidence.

That completion condition is now satisfied: 27/27 unique formal cells are
valid, all nine trace groups pair exactly, warmups and drains are complete, and
the frozen final status is `FAIL`.

Do not commit model weights, Adapter `.safetensors`, or detailed request-level
artifacts. Raw artifacts stay on the WSL2 host. Commit only the compact reports
listed by the E08 runbook.

## 10. Important code and document map

- `docs/E07_PROTOCOL.md`: research question, frozen protocol, and gates.
- `docs/E07_DATA_CARD.md`: generated training/validation/test data provenance.
- `docs/M3_E07_QLORA_LORA_RUNBOOK.md`: authoritative manual GPU sequence.
- `docs/E07_RESULTS.md`: measured result, deployment decision, and limitations.
- `reports/e07_lora/readiness.md`: pre-GPU audit status.
- `configs/train/e07_*.toml`: smoke, rank-8, and rank-16 training configs.
- `configs/serve/e07_base.toml`: Base serving profile.
- `configs/serve/e07_lora.toml`: primary rank-8 LoRA serving profile.
- `configs/matrix/e07_*.toml`: frozen Base/LoRA performance matrices.
- `src/qwen_serve_lab/e07*.py`: data, training, inspection, quality, comparison,
  blind-review, readiness, and finalization logic.
- `docs/M4_E08_ADMISSION_RUNBOOK.md`: authoritative E08 protocol and GPU order.
- `docs/E08_RESULTS.md`: measured E08 result, deployment decision, and limits.
- `configs/admission/e08.toml`: frozen traces, policies, AIMD, and gates.
- `src/qwen_serve_lab/e08*.py`: admission, runner, comparison, and readiness.
- `reports/e08_admission/comparison.md` and `final.json`: completed E08 result.
- `reports/e08_admission/readiness.md`: retained pre-GPU audit, not the result.
- `docs/E09_PROTOCOL.md`: independent holdout question, calibration boundary,
  frozen queue controller, and positive-result gates.
- `docs/M4_E09_DEADLINE_ADMISSION_RUNBOOK.md`: authoritative E09 GPU sequence.
- `configs/admission/e09.toml`: disjoint seeds, burst holdouts, C8/8192-token/
  800-ms deadline controller, and gates.
- `src/qwen_serve_lab/e09*.py`: trace, runner, comparison, and readiness logic.
- `docs/E09_RESULTS.md`: measured positive result, deployment boundary, and
  interpretation.
- `reports/e09_deadline_admission/comparison.md` and `final.json`: completed E09
  result and frozen `PASS`.
- `reports/e09_deadline_admission/readiness.md`: retained pre-GPU audit, not the
  final result.
- `Makefile`: operator-facing entry points.

Before editing, inspect `git status`, the latest commit, relevant tests, and the
latest user output. Work with existing user changes and never reset or clean a
dirty worktree automatically.

## 11. New-Agent startup prompt

The user can start the next Codex task with:

```text
继续 QwenServe-12G 项目。请先阅读 docs/CODEX_AGENT_HANDOFF.md，检查当前
git status、分支和最新提交。E01-E09 已执行完成；E07 在线成本总体 FAIL，E08
冻结科学门槛总体 FAIL，E09 deadline-aware admission 的 27/27 formal cells
有效且最终 PASS，periodic/shock burst 中位 SLO-goodput 提升 +18.75%/+17.43%。
原始 E09 artifacts 保留在 WSL2 的 ~/projects/QwenServe-12G-e09，Git 只提交紧凑
报告。不要触碰 reports/e05_kv_cache/human_review.backup.csv，也不要把 E09 的
burst 结论泛化到 steady overload 或其他硬件/模型。
```
