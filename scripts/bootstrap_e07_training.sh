#!/usr/bin/env bash
set -euo pipefail

if [[ ! -x .venv/bin/python ]]; then
  echo "Run scripts/bootstrap_wsl.sh before installing E07 dependencies." >&2
  exit 1
fi
if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required." >&2
  exit 1
fi

mkdir -p artifacts/env
uv pip install \
  --python .venv/bin/python \
  --no-deps \
  --constraint constraints/e07-train.txt \
  accelerate bitsandbytes peft
.venv/bin/python -c 'import accelerate, bitsandbytes, peft; print("E07 training dependencies: PASS")'
.venv/bin/python -m pip check
uv pip freeze --python .venv/bin/python > artifacts/env/e07-training-freeze.txt
