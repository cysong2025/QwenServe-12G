PYTHON ?= python3
QSL = PYTHONPATH=src $(PYTHON) -m qwen_serve_lab.cli
MODEL_PATH ?= $(HOME)/models/Qwen2.5-3B-Instruct

.PHONY: doctor collect-env repair-bench-deps download-model-modelscope render-baseline render-baseline-local serve-baseline serve-baseline-local render-prefix render-baseline-matrix bench-smoke bench-baseline bench-baseline-matrix summarize-pilot summarize test

doctor:
	$(QSL) doctor

collect-env:
	$(QSL) collect-env --output artifacts/env/manual.json

repair-bench-deps:
	uv pip install --python .venv/bin/python --constraint constraints/vllm-0.25.1.txt pyarrow
	.venv/bin/vllm bench serve --help >/dev/null
	uv pip freeze --python .venv/bin/python > artifacts/env/bootstrap-freeze.txt

download-model-modelscope:
	bash scripts/download_model_modelscope.sh "$(MODEL_PATH)"

render-baseline:
	$(QSL) render-serve configs/serve/baseline.toml

render-baseline-local:
	$(QSL) render-serve configs/serve/baseline.toml --model-path "$(MODEL_PATH)"

serve-baseline:
	$(QSL) run-serve configs/serve/baseline.toml

serve-baseline-local:
	$(QSL) run-serve configs/serve/baseline.toml --model-path "$(MODEL_PATH)"

render-prefix:
	$(QSL) render-serve configs/serve/prefix_cache.toml

render-baseline-matrix:
	$(QSL) render-matrix configs/matrix/baseline.toml --tokenizer-path "$(MODEL_PATH)"

bench-smoke:
	$(QSL) run-bench configs/bench/smoke.toml --tokenizer-path "$(MODEL_PATH)"

bench-baseline:
	$(QSL) run-matrix configs/matrix/baseline.toml --only e01_baseline_short_c1 --tokenizer-path "$(MODEL_PATH)"

bench-baseline-matrix:
	$(QSL) run-matrix configs/matrix/baseline.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

summarize-pilot:
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/pilot --profile-prefix e01_baseline --benchmark-config configs/bench/baseline_short_c1.toml

summarize:
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/baseline --profile-prefix e01_baseline --benchmark-config configs/matrix/baseline.toml

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
