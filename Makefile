PYTHON ?= python3
QSL = PYTHONPATH=src $(PYTHON) -m qwen_serve_lab.cli

.PHONY: doctor collect-env render-baseline serve-baseline render-prefix render-baseline-matrix bench-smoke bench-baseline bench-baseline-matrix summarize test

doctor:
	$(QSL) doctor

collect-env:
	$(QSL) collect-env --output artifacts/env/manual.json

render-baseline:
	$(QSL) render-serve configs/serve/baseline.toml

serve-baseline:
	$(QSL) run-serve configs/serve/baseline.toml

render-prefix:
	$(QSL) render-serve configs/serve/prefix_cache.toml

render-baseline-matrix:
	$(QSL) render-matrix configs/matrix/baseline.toml

bench-smoke:
	$(QSL) run-bench configs/bench/smoke.toml

bench-baseline:
	$(QSL) run-bench configs/bench/baseline_short_c1.toml

bench-baseline-matrix:
	$(QSL) run-matrix configs/matrix/baseline.toml

summarize:
	$(QSL) summarize --manifest-dir artifacts/env --output-dir reports/baseline --profile-prefix e01_baseline

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v
