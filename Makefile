PYTHON ?= .venv/bin/python
PYTEST_ARGS ?=

.PHONY: test test-physics test-validation validate-external bench bench-extended lint clean

## Run physics unit tests and validation tests (~5 min, includes example sims).
test: test-physics test-validation

## Core test suite excluding long-running telemetry validation.
test-physics:
	$(PYTHON) -m pytest tests --ignore=tests/test_validation.py $(PYTEST_ARGS)

## Physics + LightSail 2 validation (numpy + sgp4; ~1 s).
test-validation:
	$(PYTHON) -m pytest tests/test_validation.py $(PYTEST_ARGS)

## Full LightSail 2 external-data validation (requires validation/data/* mission files).
validate-external:
	$(PYTHON) -c "from pathlib import Path; import sys; from cislunar.validation import DEFAULT_BEACON_FILE, DEFAULT_LIGHTSAIL2_ARCHIVE; missing = [path for path in (DEFAULT_LIGHTSAIL2_ARCHIVE, DEFAULT_BEACON_FILE) if not Path(path).exists()]; [print(f'Missing external validation file: {path}') for path in missing]; sys.exit(1 if missing else 0)"
	$(PYTHON) -m pytest tests/test_validation.py $(PYTEST_ARGS)

## External physics benchmarks against published reference values (McInnes, Wertz, JPL Horizons).
bench:
	$(PYTHON) -m cislunar.validation.physics_benchmark

## Full benchmark suite (19 benchmarks).
bench-extended:
	$(PYTHON) -m cislunar.validation.physics_benchmark --extended

## Lint and format with ruff.
lint:
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

## Remove build artifacts and caches.
clean:
	find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} +
	find . -type f -name '*.pyc' -not -path './.venv/*' -delete
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov dist build src/*.egg-info
