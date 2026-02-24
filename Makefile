.PHONY: test test-v test-cov test-cov-html lint format install clean pipeline \
        docker-build-api docker-build-dashboard docker-build-all \
        docker-run-api docker-run-dashboard \
        compose-up compose-up-d compose-down compose-logs

VENV := .venv
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest

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
	$(PYTHON) -m src.train --xls "$(XLS)" --model-dir "$(MODEL_DIR)"

# ── Docker / HF Spaces ────────────────────────────────────────────────────────

docker-build-api:
	docker build -f Dockerfile.api -t pm-api:latest .

docker-build-dashboard:
	docker build -f Dockerfile.dashboard -t pm-dashboard:latest .

docker-build-all: docker-build-api docker-build-dashboard

docker-run-api:
	docker run --rm -p 7860:7860 pm-api:latest

docker-run-dashboard:
	docker run --rm -p 7861:7860 pm-dashboard:latest

# ── Docker Compose (ambos os serviços juntos) ─────────────────────────────────

compose-up:
	docker compose up --build

compose-up-d:
	docker compose up --build -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f
