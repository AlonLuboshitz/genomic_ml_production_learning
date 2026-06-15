# Makefile for the genomics ML pipeline
#
# Usage:
#   make install    — install the package in editable mode
#   make test       — run all unit tests
#   make train      — train the baseline model
#   make smoke-test — quick end-to-end check
#   make format     — format code with ruff
#   make lint       — lint with ruff
#   make evaluate   — evaluate a trained model
#   make serve      — run the FastAPI server
#   make run-pipeline — run the full Prefect pipeline
#   make docker-build — build Docker image
#   make docker-up   — start Docker Compose services

# Use the project virtual environment if it exists, otherwise fall back to system Python
PYTHON := $(shell [ -f .venv/bin/python ] && echo ".venv/bin/python" || echo "python")
PIP    := $(shell [ -f .venv/bin/pip ] && echo ".venv/bin/pip" || echo "pip")
RUFF   := $(shell [ -f .venv/bin/ruff ] && echo ".venv/bin/ruff" || echo "ruff")

.PHONY: install test train smoke-test format lint evaluate serve run-pipeline docker-build docker-up

install:
	$(PIP) install -e ".[dev]"

test:
	$(PYTHON) -m pytest tests/ -v

train:
	$(PYTHON) scripts/train_model.py

smoke-test:
	$(PYTHON) -c "from genomics_ml.models.train import train_model; from genomics_ml.data.load_data import load_data; X, y = load_data(); metrics, _ = train_model(X, y); print('Smoke test passed — accuracy:', metrics['accuracy'])"

format:
	$(RUFF) format src/ tests/ scripts/

lint:
	$(RUFF) check src/ tests/ scripts/

evaluate:
	$(PYTHON) scripts/evaluate_model.py

serve:
	$(PYTHON) -m uvicorn genomics_ml.api.main:app --reload --port 8000

run-pipeline:
	@echo "TODO: implement run-pipeline target"

docker-build:
	docker build -t genomics-ml-api .

docker-up:
	docker compose up
