#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-3B-Instruct"
MODEL_DIR="${1:-${HOME}/models/Qwen2.5-3B-Instruct}"

if ! command -v uvx >/dev/null 2>&1; then
  echo "uvx is required. Install uv and rerun this script." >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"
uvx --from modelscope modelscope download \
  --local_dir "${MODEL_DIR}" \
  "${MODEL_ID}"

for required in config.json tokenizer.json; do
  if [[ ! -f "${MODEL_DIR}/${required}" ]]; then
    echo "Downloaded snapshot is missing ${required}." >&2
    exit 1
  fi
done

if ! compgen -G "${MODEL_DIR}/*.safetensors" >/dev/null; then
  echo "Downloaded snapshot contains no safetensors weights." >&2
  exit 1
fi

echo "Computing SHA-256 manifest for the local model snapshot..."
(
  cd "${MODEL_DIR}"
  find . -maxdepth 1 -type f \
    -not -name SHA256SUMS \
    -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 sha256sum > SHA256SUMS
)

echo "Model snapshot: ${MODEL_DIR}"
echo "SHA-256 manifest: ${MODEL_DIR}/SHA256SUMS"
