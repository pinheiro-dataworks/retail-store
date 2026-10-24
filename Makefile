# Common tasks. Run `make help` for the list.
PYTHON ?= python
export PYTHONPATH := src

.PHONY: help setup data train train-quick test lint format serve check

help:
	@echo "setup        install pinned runtime + dev dependencies"
	@echo "data         download the raw CSVs (set RETAIL_DATASET=<owner>/<slug>) and verify them"
	@echo "train        full run: tuning, holdout evaluation, forecast, export to web/data/"
	@echo "train-quick  same without the hyperparameter search"
	@echo "test         python and node test suites"
	@echo "lint         ruff"
	@echo "serve        static preview of the dashboard on http://localhost:8080"

setup:
	$(PYTHON) -m pip install -r requirements-dev.txt

data:
	$(PYTHON) scripts/download_data.py $(if $(RETAIL_DATASET),--dataset $(RETAIL_DATASET),--verify-only)

train:
	$(PYTHON) -m retail_forecast

train-quick:
	$(PYTHON) -m retail_forecast --quick

test:
	$(PYTHON) -m pytest

lint:
	ruff check src tests scripts

format:
	ruff check --fix src tests scripts

serve:
	cd web && $(PYTHON) -m http.server 8080

check: lint test
