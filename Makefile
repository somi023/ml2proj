# Uses the repo's .venv when it exists (laptop); otherwise the `python` on PATH (Colab, iLab job, CI).
PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python)

.PHONY: setup test lint smoke reproduce

setup:  ## pinned runtime + dev packages, and the git pre-commit hook (once per clone)
	$(PYTHON) -m pip install -r requirements-dev.txt
	$(PYTHON) -m pre_commit install

test:  ## CPU unit tests; GPU tests carry the gpu marker and run by hand
	$(PYTHON) -m pytest -m "not gpu"

lint:  ## lint and format check; changes nothing
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

smoke:  ## whole pipeline on 32 items, under 2 minutes, CPU allowed
	@echo "make smoke: src/run.py --smoke does not exist yet (plan Day 2)"; exit 1

reproduce:  ## rebuild every figure and table from results/ without a GPU
	@echo "make reproduce: src/figures.py does not exist yet"; exit 1
