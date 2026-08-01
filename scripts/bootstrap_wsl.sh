#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
VLLM_VERSION="${VLLM_VERSION:-0.25.1}"

if ! grep -qi microsoft /proc/version; then
  echo "This bootstrap script must run inside WSL2." >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is unavailable. Update the Windows NVIDIA driver; do not install a Linux display driver in WSL." >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required. Install it from https://docs.astral.sh/uv/ and rerun this script." >&2
  exit 1
fi

nvidia-smi
uv venv --python "${PYTHON_VERSION}" --seed .venv
source .venv/bin/activate
uv pip install \
  --constraint constraints/vllm-0.25.1.txt \
  "vllm[bench]==${VLLM_VERSION}" \
  --torch-backend=auto
uv pip install --editable .
uv pip freeze > artifacts/env/bootstrap-freeze.txt
python -m qwen_serve_lab.cli doctor
