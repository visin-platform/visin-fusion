# Every check CI runs, runnable here. `make check` is the one to run before a PR.
#
# Uses venv/ when it exists, the PATH otherwise, which is how CI runs the same targets.

VENV ?= venv
BIN := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/,)
PYTHON ?= python3
MODEL ?= clftv2
MODES ?= fusion

.DEFAULT_GOAL := help
.PHONY: help install lint typecheck format test e2e coverage docs check clean

help: ## List the targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-9s %s\n", $$1, $$2}'

install: ## Create venv with every dev tool, and install the git hooks
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -e '.[visin,dev]'
	$(VENV)/bin/pre-commit install

lint: ## Lint and check formatting (ruff)
	$(BIN)ruff check .
	$(BIN)ruff format --check .

typecheck: ## Type-check the annotated modules (mypy)
	$(BIN)mypy

format: ## Fix formatting and the lint findings ruff can fix
	$(BIN)ruff format .
	$(BIN)ruff check --fix .

test: ## Run the unit tests
	$(BIN)pytest tests/unit

e2e: ## End-to-end on the sample dataset, CPU: make e2e MODEL=clftv2 MODES=fusion
	$(BIN)pytest tests/e2e --models $(MODEL) --modes $(MODES) --device cpu --visin offline

coverage: ## Unit + fusion end-to-end coverage, failing under the floor
	$(BIN)python tools/coverage.py

docs: ## Build the docs site, failing on any warning
	$(BIN)python tools/make_config_reference.py --check
	$(BIN)python tools/generate_model_diagrams.py --check
	$(BIN)mkdocs build --strict

check: lint typecheck test docs ## What CI checks, without the end-to-end runs
	@echo "all checks passed"

clean: ## Remove build, test and docs output
	rm -rf build dist site .coverage .coverage.* .pytest_cache .ruff_cache
