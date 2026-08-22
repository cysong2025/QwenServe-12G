PYTHON ?= python3
QSL = PYTHONPATH=src $(PYTHON) -m qwen_serve_lab.cli
MODEL_PATH ?= $(HOME)/models/Qwen2.5-3B-Instruct
E02_BUDGET ?= 8192
E02_SERVE_CONFIG = configs/serve/e02_batch_tokens_$(E02_BUDGET).toml
E02_MATRIX_CONFIG = configs/matrix/e02_batch_tokens_$(E02_BUDGET).toml
E07_ADAPTER_PATH ?= artifacts/adapters/e07/rank8

.PHONY: doctor collect-env repair-bench-deps download-model-modelscope render-baseline render-baseline-local serve-baseline serve-baseline-local render-prefix render-baseline-matrix bench-smoke bench-baseline bench-baseline-matrix summarize-pilot summarize render-e02 render-e02-local serve-e02 serve-e02-local render-e02-matrix bench-e02-pilot bench-e02-matrix summarize-e02 compare-e02 render-e04-off-local render-e04-on-local serve-e04-off-local serve-e04-on-local render-e04-off-matrix render-e04-on-matrix bench-e04-off-pilot bench-e04-on-pilot bench-e04-off-matrix bench-e04-on-matrix bench-e04-off-capacity bench-e04-on-capacity summarize-e04 compare-e04 diagnose-e04 run-e04-canary-off run-e04-canary-on compare-e04-canary render-e05-bf16-local render-e05-fp8-local serve-e05-bf16-local serve-e05-fp8-local render-e05-bf16-matrix render-e05-fp8-matrix bench-e05-bf16-pilot bench-e05-fp8-pilot bench-e05-bf16-matrix bench-e05-fp8-matrix run-e05-quality-bf16 run-e05-quality-fp8 summarize-e05 compare-e05 capacity-e05 compare-e05-quality summarize-e05-human-review finalize-e05 test
.PHONY: render-e06-bt8192-off-local render-e06-bt2048-off-local render-e06-bt8192-on-local render-e06-bt2048-on-local serve-e06-bt8192-off-local serve-e06-bt2048-off-local serve-e06-bt8192-on-local serve-e06-bt2048-on-local render-e06-bt8192-off-matrix render-e06-bt2048-off-matrix render-e06-bt8192-on-matrix render-e06-bt2048-on-matrix bench-e06-bt8192-off-pilot bench-e06-bt2048-off-pilot bench-e06-bt8192-on-pilot bench-e06-bt2048-on-pilot bench-e06-bt8192-off-matrix bench-e06-bt2048-off-matrix bench-e06-bt8192-on-matrix bench-e06-bt2048-on-matrix run-e06-canary-bt8192-off run-e06-canary-bt2048-off run-e06-canary-bt8192-on run-e06-canary-bt2048-on summarize-e06 compare-e06 compare-e06-canary audit-e01-e06
.PHONY: install-e07-train-deps prepare-e07-data audit-e07-readiness render-e07-smoke train-e07-smoke train-e07-rank8 train-e07-rank16 inspect-e07-adapter render-e07-base-local render-e07-lora-local serve-e07-base-local serve-e07-lora-local render-e07-base-matrix render-e07-lora-matrix bench-e07-base-pilot bench-e07-lora-pilot bench-e07-base-matrix bench-e07-lora-matrix run-e07-quality-base run-e07-quality-lora summarize-e07 compare-e07 compare-e07-quality summarize-e07-human-review finalize-e07

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

finalize-e05:
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/e05_kv_cache --profile-prefix e05_
	$(QSL) compare-e05
	$(QSL) capacity-e05
	$(QSL) compare-e05-quality; status=$$?; test $$status -eq 0 -o $$status -eq 2
	$(QSL) summarize-e05-human-review; status=$$?; test $$status -eq 0 -o $$status -eq 2
	test "$$(($$(wc -l < reports/e05_kv_cache/runs.csv) - 1))" -eq 36
	test -f reports/e05_kv_cache/comparison.csv
	test -f reports/e05_kv_cache/capacity.json
	test -f reports/e05_kv_cache/quality.json
	test -f reports/e05_kv_cache/human_review_summary.json

render-e06-bt8192-off-local:
	$(QSL) render-serve configs/serve/e06_bt8192_apc_off.toml --model-path "$(MODEL_PATH)"

render-e06-bt2048-off-local:
	$(QSL) render-serve configs/serve/e06_bt2048_apc_off.toml --model-path "$(MODEL_PATH)"

render-e06-bt8192-on-local:
	$(QSL) render-serve configs/serve/e06_bt8192_apc_on.toml --model-path "$(MODEL_PATH)"

render-e06-bt2048-on-local:
	$(QSL) render-serve configs/serve/e06_bt2048_apc_on.toml --model-path "$(MODEL_PATH)"

serve-e06-bt8192-off-local:
	$(QSL) run-serve configs/serve/e06_bt8192_apc_off.toml --model-path "$(MODEL_PATH)"

serve-e06-bt2048-off-local:
	$(QSL) run-serve configs/serve/e06_bt2048_apc_off.toml --model-path "$(MODEL_PATH)"

serve-e06-bt8192-on-local:
	$(QSL) run-serve configs/serve/e06_bt8192_apc_on.toml --model-path "$(MODEL_PATH)"

serve-e06-bt2048-on-local:
	$(QSL) run-serve configs/serve/e06_bt2048_apc_on.toml --model-path "$(MODEL_PATH)"

render-e06-bt8192-off-matrix:
	$(QSL) render-matrix configs/matrix/e06_bt8192_apc_off.toml --tokenizer-path "$(MODEL_PATH)"
	$(QSL) render-matrix configs/matrix/e06_bt8192_apc_off_capacity.toml --tokenizer-path "$(MODEL_PATH)"

render-e06-bt2048-off-matrix:
	$(QSL) render-matrix configs/matrix/e06_bt2048_apc_off.toml --tokenizer-path "$(MODEL_PATH)"
	$(QSL) render-matrix configs/matrix/e06_bt2048_apc_off_capacity.toml --tokenizer-path "$(MODEL_PATH)"

render-e06-bt8192-on-matrix:
	$(QSL) render-matrix configs/matrix/e06_bt8192_apc_on.toml --tokenizer-path "$(MODEL_PATH)"
	$(QSL) render-matrix configs/matrix/e06_bt8192_apc_on_capacity.toml --tokenizer-path "$(MODEL_PATH)"

render-e06-bt2048-on-matrix:
	$(QSL) render-matrix configs/matrix/e06_bt2048_apc_on.toml --tokenizer-path "$(MODEL_PATH)"
	$(QSL) render-matrix configs/matrix/e06_bt2048_apc_on_capacity.toml --tokenizer-path "$(MODEL_PATH)"

bench-e06-bt8192-off-pilot:
	$(QSL) run-matrix configs/matrix/e06_bt8192_apc_off.toml --only e06_bt8192_off_reuse90_p1024_c4 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e06-bt2048-off-pilot:
	$(QSL) run-matrix configs/matrix/e06_bt2048_apc_off.toml --only e06_bt2048_off_reuse90_p1024_c4 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e06-bt8192-on-pilot:
	$(QSL) run-matrix configs/matrix/e06_bt8192_apc_on.toml --only e06_bt8192_on_reuse90_p1024_c4 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e06-bt2048-on-pilot:
	$(QSL) run-matrix configs/matrix/e06_bt2048_apc_on.toml --only e06_bt2048_on_reuse90_p1024_c4 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e06-bt8192-off-matrix:
	$(QSL) run-matrix configs/matrix/e06_bt8192_apc_off.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed
	$(QSL) run-matrix configs/matrix/e06_bt8192_apc_off_capacity.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e06-bt2048-off-matrix:
	$(QSL) run-matrix configs/matrix/e06_bt2048_apc_off.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed
	$(QSL) run-matrix configs/matrix/e06_bt2048_apc_off_capacity.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e06-bt8192-on-matrix:
	$(QSL) run-matrix configs/matrix/e06_bt8192_apc_on.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed
	$(QSL) run-matrix configs/matrix/e06_bt8192_apc_on_capacity.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e06-bt2048-on-matrix:
	$(QSL) run-matrix configs/matrix/e06_bt2048_apc_on.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed
	$(QSL) run-matrix configs/matrix/e06_bt2048_apc_on_capacity.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

run-e06-canary-bt8192-off:
	$(QSL) run-e06-canary --state bt8192_off

run-e06-canary-bt2048-off:
	$(QSL) run-e06-canary --state bt2048_off

run-e06-canary-bt8192-on:
	$(QSL) run-e06-canary --state bt8192_on

run-e06-canary-bt2048-on:
	$(QSL) run-e06-canary --state bt2048_on

summarize-e06:
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/e06_combined --profile-prefix e06_

compare-e06:
	$(QSL) compare-e06

compare-e06-canary:
	$(QSL) compare-e06-canary

audit-e01-e06:
	$(QSL) audit-e01-e06

install-e07-train-deps:
	bash scripts/bootstrap_e07_training.sh

prepare-e07-data:
	$(QSL) prepare-e07-data

audit-e07-readiness:
	$(QSL) audit-e07-readiness

render-e07-smoke:
	$(QSL) render-e07-train configs/train/e07_qlora_smoke_rank8.toml --model-path "$(MODEL_PATH)"

train-e07-smoke:
	$(QSL) train-e07 configs/train/e07_qlora_smoke_rank8.toml --model-path "$(MODEL_PATH)"

train-e07-rank8:
	$(QSL) train-e07 configs/train/e07_qlora_rank8.toml --model-path "$(MODEL_PATH)"

train-e07-rank16:
	$(QSL) train-e07 configs/train/e07_qlora_rank16.toml --model-path "$(MODEL_PATH)"

inspect-e07-adapter:
	$(QSL) inspect-e07-adapter --adapter-dir "$(E07_ADAPTER_PATH)" --expected-rank 8

render-e07-base-local:
	$(QSL) render-serve configs/serve/e07_base.toml --model-path "$(MODEL_PATH)"

render-e07-lora-local:
	$(QSL) render-serve configs/serve/e07_lora.toml --model-path "$(MODEL_PATH)"

serve-e07-base-local:
	$(QSL) run-serve configs/serve/e07_base.toml --model-path "$(MODEL_PATH)"

serve-e07-lora-local:
	$(QSL) run-serve configs/serve/e07_lora.toml --model-path "$(MODEL_PATH)" --adapter-path "$(E07_ADAPTER_PATH)"

render-e07-base-matrix:
	$(QSL) render-matrix configs/matrix/e07_base.toml --tokenizer-path "$(MODEL_PATH)"

render-e07-lora-matrix:
	$(QSL) render-matrix configs/matrix/e07_lora.toml --tokenizer-path "$(MODEL_PATH)"

bench-e07-base-pilot:
	$(QSL) run-matrix configs/matrix/e07_base.toml --only e07_base_medium_c4 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e07-lora-pilot:
	$(QSL) run-matrix configs/matrix/e07_lora.toml --only e07_lora_medium_c4 --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e07-base-matrix:
	$(QSL) run-matrix configs/matrix/e07_base.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

bench-e07-lora-matrix:
	$(QSL) run-matrix configs/matrix/e07_lora.toml --tokenizer-path "$(MODEL_PATH)" --skip-completed

run-e07-quality-base:
	$(QSL) run-e07-quality --state base

run-e07-quality-lora:
	$(QSL) run-e07-quality --state lora

summarize-e07:
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/e07_lora --profile-prefix e07_

compare-e07:
	$(QSL) compare-e07

compare-e07-quality:
	$(QSL) compare-e07-quality

summarize-e07-human-review:
	$(QSL) summarize-e07-human-review

finalize-e07:
	$(QSL) inspect-e07-adapter --adapter-dir artifacts/adapters/e07/rank8 --expected-rank 8
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/e07_lora --profile-prefix e07_
	$(QSL) compare-e07
	$(QSL) compare-e07-quality
	$(QSL) summarize-e07-human-review
	$(QSL) finalize-e07

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
