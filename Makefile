.PHONY: install lint test eda train evaluate api ui

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/config.yaml
API_HOST ?= 0.0.0.0
API_PORT ?= 8000

install:
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

eda:
	$(PYTHON) -m src.eda.run_eda --config $(CONFIG)

train:
	$(PYTHON) -m src.training.train --config $(CONFIG)

evaluate:
	$(PYTHON) -m src.evaluation.evaluate --config $(CONFIG)

api:
	$(PYTHON) -m uvicorn app.api.main:app --host $(API_HOST) --port $(API_PORT)

ui:
	$(PYTHON) -m streamlit run app/ui/streamlit_app.py
