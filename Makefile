PYTHON ?= python3
QSL = PYTHONPATH=src $(PYTHON) -m qwen_serve_lab.cli
MODEL_PATH ?= $(HOME)/models/Qwen2.5-3B-Instruct
E02_BUDGET ?= 8192
E02_SERVE_CONFIG = configs/serve/e02_batch_tokens_$(E02_BUDGET).toml
E02_MATRIX_CONFIG = configs/matrix/e02_batch_tokens_$(E02_BUDGET).toml

.PHONY: doctor collect-env repair-bench-deps download-model-modelscope render-baseline render-baseline-local serve-baseline serve-baseline-local render-prefix render-baseline-matrix bench-smoke bench-baseline bench-baseline-matrix summarize-pilot summarize render-e02 render-e02-local serve-e02 serve-e02-local render-e02-matrix bench-e02-pilot bench-e02-matrix summarize-e02 compare-e02 render-e04-off-local render-e04-on-local serve-e04-off-local serve-e04-on-local render-e04-off-matrix render-e04-on-matrix bench-e04-off-pilot bench-e04-on-pilot bench-e04-off-matrix bench-e04-on-matrix bench-e04-off-capacity bench-e04-on-capacity summarize-e04 compare-e04 diagnose-e04 run-e04-canary-off run-e04-canary-on compare-e04-canary render-e05-bf16-local render-e05-fp8-local serve-e05-bf16-local serve-e05-fp8-local render-e05-bf16-matrix render-e05-fp8-matrix bench-e05-bf16-pilot bench-e05-fp8-pilot bench-e05-bf16-matrix bench-e05-fp8-matrix run-e05-quality-bf16 run-e05-quality-fp8 summarize-e05 compare-e05 capacity-e05 compare-e05-quality summarize-e05-human-review test

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

render-e02:
	$(QSL) render-serve $(E02_SERVE_CONFIG)

render-e02-local:
	$(QSL) render-serve $(E02_SERVE_CONFIG) --model-path "$(MODEL_PATH)"

serve-e02:
	$(QSL) run-serve $(E02_SERVE_CONFIG)

serve-e02-local:
	$(QSL) run-serve $(E02_SERVE_CONFIG) --model-path "$(MODEL_PATH)"

render-e02-matrix:
	$(QSL) render-matrix $(E02_MATRIX_CONFIG) --tokenizer-path "$(MODEL_PATH)"

bench-e02-pilot:
	$(QSL) run-matrix $(E02_MATRIX_CONFIG) --only e02_bt$(E02_BUDGET)_medium_c8 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e02-matrix:
	$(QSL) run-matrix $(E02_MATRIX_CONFIG) --tokenizer-path "$(MODEL_PATH)" --skip-completed

summarize-e02:
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/e02_batch_tokens --profile-prefix e02_bt

compare-e02:
	$(QSL) compare-e02 --runs-csv reports/e02_batch_tokens/runs.csv --output-dir reports/e02_batch_tokens --reference-budget 8192

render-e04-off-local:
	$(QSL) render-serve configs/serve/e04_prefix_off.toml --model-path "$(MODEL_PATH)"

render-e04-on-local:
	$(QSL) render-serve configs/serve/e04_prefix_on.toml --model-path "$(MODEL_PATH)"

serve-e04-off-local:
	$(QSL) run-serve configs/serve/e04_prefix_off.toml --model-path "$(MODEL_PATH)"

serve-e04-on-local:
	$(QSL) run-serve configs/serve/e04_prefix_on.toml --model-path "$(MODEL_PATH)"

render-e04-off-matrix:
	$(QSL) render-matrix configs/matrix/e04_prefix_off.toml --tokenizer-path "$(MODEL_PATH)"
	$(QSL) render-matrix configs/matrix/e04_prefix_off_capacity.toml --tokenizer-path "$(MODEL_PATH)"

render-e04-on-matrix:
	$(QSL) render-matrix configs/matrix/e04_prefix_on.toml --tokenizer-path "$(MODEL_PATH)"
	$(QSL) render-matrix configs/matrix/e04_prefix_on_capacity.toml --tokenizer-path "$(MODEL_PATH)"

bench-e04-off-pilot:
	$(QSL) run-matrix configs/matrix/e04_prefix_off.toml --only e04_off_reuse90_p1024_c4 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e04-on-pilot:
	$(QSL) run-matrix configs/matrix/e04_prefix_on.toml --only e04_on_reuse90_p1024_c4 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e04-off-matrix:
	$(QSL) run-matrix configs/matrix/e04_prefix_off.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e04-on-matrix:
	$(QSL) run-matrix configs/matrix/e04_prefix_on.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e04-off-capacity:
	$(QSL) run-matrix configs/matrix/e04_prefix_off_capacity.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e04-on-capacity:
	$(QSL) run-matrix configs/matrix/e04_prefix_on_capacity.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

summarize-e04:
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/e04_prefix_cache --profile-prefix e04_

compare-e04:
	$(QSL) compare-e04 --runs-csv reports/e04_prefix_cache/runs.csv --output-dir reports/e04_prefix_cache

diagnose-e04:
	$(QSL) diagnose-e04 --runs-csv reports/e04_prefix_cache/runs.csv --output-dir reports/e04_prefix_cache

run-e04-canary-off:
	$(QSL) run-e04-canary --state off

run-e04-canary-on:
	$(QSL) run-e04-canary --state on

compare-e04-canary:
	$(QSL) compare-e04-canary

render-e05-bf16-local:
	$(QSL) render-serve configs/serve/e05_kv_bf16.toml --model-path "$(MODEL_PATH)"

render-e05-fp8-local:
	$(QSL) render-serve configs/serve/e05_kv_fp8.toml --model-path "$(MODEL_PATH)"

serve-e05-bf16-local:
	$(QSL) run-serve configs/serve/e05_kv_bf16.toml --model-path "$(MODEL_PATH)"

serve-e05-fp8-local:
	$(QSL) run-serve configs/serve/e05_kv_fp8.toml --model-path "$(MODEL_PATH)"

render-e05-bf16-matrix:
	$(QSL) render-matrix configs/matrix/e05_kv_bf16.toml --tokenizer-path "$(MODEL_PATH)"

render-e05-fp8-matrix:
	$(QSL) render-matrix configs/matrix/e05_kv_fp8.toml --tokenizer-path "$(MODEL_PATH)"

bench-e05-bf16-pilot:
	$(QSL) run-matrix configs/matrix/e05_kv_bf16.toml --only e05_bf16_long_c8 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e05-fp8-pilot:
	$(QSL) run-matrix configs/matrix/e05_kv_fp8.toml --only e05_fp8_long_c8 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e05-bf16-matrix:
	$(QSL) run-matrix configs/matrix/e05_kv_bf16.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e05-fp8-matrix:
	$(QSL) run-matrix configs/matrix/e05_kv_fp8.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

run-e05-quality-bf16:
	$(QSL) run-e05-quality --state bf16

run-e05-quality-fp8:
	$(QSL) run-e05-quality --state fp8

summarize-e05:
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/e05_kv_cache --profile-prefix e05_

compare-e05:
	$(QSL) compare-e05

capacity-e05:
	$(QSL) capacity-e05

compare-e05-quality:
	$(QSL) compare-e05-quality

summarize-e05-human-review:
	$(QSL) summarize-e05-human-review

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
