# M4 E09 Deadline-aware Admission Runbook

This runbook executes E09 on the controlled Windows/WSL2 RTX 5070 host. E08 is
retained as a negative result. E09 is not complete until the full formal matrix
and frozen comparison report exist.

## 1. Mac preflight and push

```bash
cd "/Users/songchuangye/Documents/推理训练"
git switch codex/e09-deadline-admission
git status --short --branch
python3 -m unittest discover -s tests -v
PYTHONPATH=src python3 -m qwen_serve_lab.cli audit-e09-readiness \
  --config configs/admission/e09.toml
git push -u origin codex/e09-deadline-admission
```

The readiness report must say `READY_FOR_GPU`, `GPU execution: DEFERRED`, and
`Scientific result: NOT_RUN`.

## 2. WSL checkout and trace freeze

```bash
cd ~/projects
git clone --branch codex/e09-deadline-admission \
  git@github.com:cysong2025/QwenServe-12G.git QwenServe-12G-e09
cd ~/projects/QwenServe-12G-e09
ln -s ../QwenServe-12G/.venv .venv
source .venv/bin/activate
git status --short --branch
git rev-parse HEAD
make prepare-e09-traces
make audit-e09-readiness
sed -n '1,240p' reports/e09_deadline_admission/readiness.md
```

If the checkout or `.venv` already exists, do not overwrite it. Verify its
branch, commit, config hash, and cleanliness first.

## 3. Start the controlled server (terminal A)

```bash
cd ~/projects/QwenServe-12G-e09
source .venv/bin/activate
export MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"
make serve-e09-local
```

Keep terminal A open. The active marker must bind the E06 Base, BF16 KV,
batch-token 2048, APC-off server config.

## 4. Calibration burst (terminal B)

```bash
cd ~/projects/QwenServe-12G-e09
source .venv/bin/activate
export MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"
make bench-e09-calibration-unbounded
sleep 30
make bench-e09-calibration-fixed
sleep 30
make bench-e09-calibration-deadline
```

Inspect all three calibration JSON files. Each must be `valid: true`; the
deadline-aware run must keep P95 TTFT/TPOT within 1000/50 ms and all admitted
queue waits within 800 ms. Calibration output is not scientific completion.

If revision A is changed after calibration, stop here. Commit the new config and
code on Mac, regenerate every trace under the new hash, and repeat readiness.
Never mix config hashes.

## 5. Formal holdout matrix

Only after the freeze boundary is accepted:

```bash
cd ~/projects/QwenServe-12G-e09
source .venv/bin/activate
export MODEL_PATH="$HOME/models/Qwen2.5-3B-Instruct"
make bench-e09-matrix
```

Resume without overwriting preserved cells:

```bash
make bench-e09-matrix-resume
```

The matrix contains 27 formal cells. A non-valid cell stops the runner and must
be diagnosed; it must not be silently replaced.

## 6. Frozen comparison

```bash
make compare-e09; status=$?; test "$status" -eq 0 -o "$status" -eq 2
sed -n '1,260p' reports/e09_deadline_admission/comparison.md
python -m json.tool reports/e09_deadline_admission/final.json | sed -n '1,300p'
```

Exit 0 and `status: PASS` mean the frozen positive gate passed. Exit 2 and
`status: FAIL` are a valid negative result, not project completion.

## 7. Evidence returned to Mac

Commit compact reports and protocol metadata only after inspection:

- `reports/e09_deadline_admission/readiness.json` and `.md`;
- `reports/e09_deadline_admission/runs.csv`;
- `reports/e09_deadline_admission/comparison.csv` and `.md`;
- `reports/e09_deadline_admission/final.json`.

Keep request-level run JSON, traces, and telemetry on WSL under
`artifacts/results/e09_deadline_admission/`. Do not touch
`reports/e05_kv_cache/human_review.backup.csv`.
