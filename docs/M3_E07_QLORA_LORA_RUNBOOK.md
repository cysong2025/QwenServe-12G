# E07 QLoRA and LoRA Serving Runbook

This is the exact operator sequence for the RTX 5070 WSL2 host. Training and
serving never run together. Commands marked as GPU work are intentionally not
executed during code preparation.

## 1. Update and validate

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

Continue only when the readiness status is `READY_FOR_GPU`.

## 2. Install training dependencies

```bash
make install-e07-train-deps
python -c 'import accelerate, bitsandbytes, peft; print(accelerate.__version__, bitsandbytes.__version__, peft.__version__)'
```

The installation writes `artifacts/env/e07-training-freeze.txt`. Do not perform
an unrelated environment upgrade between Base and LoRA experiments.

## 3. QLoRA smoke

Ensure no model server is alive:

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

Load the smoke Adapter in terminal 1:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e07-lora-local E07_ADAPTER_PATH=artifacts/adapters/e07/smoke-r8
```

After `Application startup complete`, verify from terminal 2:

```bash
curl -s http://127.0.0.1:8000/v1/models | python -m json.tool
```

The response must list `ai-infra-triage-r8`. Stop terminal 1 with `Ctrl+C` before
formal training.

## 4. Formal training

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
pgrep -af 'vllm serve' || true
nvidia-smi

make train-e07-rank8
make inspect-e07-adapter
sed -n '1,160p' reports/e07_lora/adapter.md
```

The rank-16 secondary ablation is run after rank 8 and does not replace the
primary Adapter in the performance matrix:

```bash
make train-e07-rank16
PYTHONPATH=src python3 -m qwen_serve_lab.cli inspect-e07-adapter \
  --adapter-dir artifacts/adapters/e07/rank16 \
  --expected-rank 16 \
  --output-dir reports/e07_lora/rank16
```

## 5. Base performance and quality

Terminal 1:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e07-base-local
```

Terminal 2, after startup completes:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate

make bench-e07-base-pilot
make run-e07-quality-base
make bench-e07-base-matrix
```

Stop terminal 1 with `Ctrl+C` after all three commands finish.

## 6. LoRA performance and quality

Terminal 1:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate
make serve-e07-lora-local
```

Terminal 2, after startup completes:

```bash
cd ~/projects/QwenServe-12G
source .venv/bin/activate

make bench-e07-lora-pilot
make run-e07-quality-lora
make bench-e07-lora-matrix
```

Stop terminal 1 with `Ctrl+C`.

The pilot cell is part of the six-cell matrix. `--skip-completed` prevents valid
pilot repetitions from being run again.

## 7. Reports and blinded review

```bash
make summarize-e07
make compare-e07
make compare-e07-quality

sed -n '1,220p' reports/e07_lora/comparison.md
sed -n '1,220p' reports/e07_lora/quality.md
```

`compare-e07-quality` creates `reports/e07_lora/human_review.csv` and a separate
blind key. Score all 50 rows without reading the key, then run:

```bash
make summarize-e07-human-review
make finalize-e07
sed -n '1,220p' reports/e07_lora/final.md
```

Exit code 2 from a comparison means a frozen scientific gate failed. Preserve
the report and raw evidence; do not loosen the gate or delete the run.

## 8. Commit reproducible evidence

Raw model weights and detailed request JSON remain local. Commit the compact
reports and manifests needed to explain the result:

```bash
git add \
  reports/e07_lora/adapter.json \
  reports/e07_lora/adapter.md \
  reports/e07_lora/runs.csv \
  reports/e07_lora/summary.md \
  reports/e07_lora/comparison.csv \
  reports/e07_lora/comparison.md \
  reports/e07_lora/quality.json \
  reports/e07_lora/quality.md \
  reports/e07_lora/human_review.csv \
  reports/e07_lora/human_review_key.json \
  reports/e07_lora/human_review_summary.json \
  reports/e07_lora/human_review_summary.md \
  reports/e07_lora/final.json \
  reports/e07_lora/final.md

git commit -m "Add RTX 5070 E07 QLoRA and LoRA results"
git push origin codex/e07-lora
```

Do not commit `artifacts/adapters`, raw detailed benchmark JSON, or model weights.
