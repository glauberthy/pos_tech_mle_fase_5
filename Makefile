.PHONY: test test-v test-cov test-cov-html lint format install clean pipeline

VENV := .venv
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

# ── Instalação ────────────────────────────────────────────────────────────────

install:
	$(PYTHON) -m pip install -e .

# ── Testes ────────────────────────────────────────────────────────────────────

test:
	$(PYTEST) tests/ -q

test-v:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ --cov=src --cov-report=term-missing --cov-fail-under=80 -q

test-cov-html:
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-fail-under=80 -q
	@echo "Relatório em htmlcov/index.html"

test-file:
	$(PYTEST) $(FILE) -v

# ── Qualidade ─────────────────────────────────────────────────────────────────

lint:
	$(VENV)/bin/ruff check src/ tests/

format:
	$(VENV)/bin/ruff format src/ tests/

# ── Limpeza ───────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache .coverage htmlcov

# ── Pipeline ──────────────────────────────────────────────────────────────────
# Uso: make pipeline
#      make pipeline XLS="outro_arquivo.xlsx" MODEL_DIR="models/v2"

XLS       ?= data/BASE DE DADOS PEDE 2024 - DATATHON.xlsx
MODEL_DIR ?= models

pipeline:
	$(PYTHON) train.py --xls "$(XLS)" --model-dir "$(MODEL_DIR)"
