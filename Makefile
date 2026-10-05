.PHONY: install lint test eda prepare train select evaluate evaluate-legacy api ui

PYTHON ?= python
PIP ?= $(PYTHON) -m pip
CONFIG ?= configs/config.yaml
MANIFESTS ?= data/processed/clean_v1
RUN_DIR ?= models/clean_runs/resnet18_clahe
SELECTED_DIR ?= models/clean_runs/selected
API_HOST ?= 127.0.0.1
API_PORT ?= 8000

install:
	$(PIP) install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest

eda:
	$(PYTHON) -m src.eda.run_eda --config $(CONFIG)

# Prepare once; repeated training uses the saved, audited manifests.
prepare:
	$(PYTHON) -m src.data.standardize_dataset --config $(CONFIG)

train:
	$(PYTHON) -m src.training.train_clean --manifests $(MANIFESTS) --model resnet18 --preprocessing clahe --cache-images --epochs 16 --lr 0.0005 --patience 5 --minimum-recall 0.98 --output $(RUN_DIR)

select:
	$(PYTHON) -m src.evaluation.evaluate_clean select --experiments $(RUN_DIR) --manifests $(MANIFESTS) --minimum-recall 0.98 --output $(SELECTED_DIR)

evaluate:
	$(PYTHON) -m src.evaluation.evaluate_clean test --manifests $(MANIFESTS) --output $(SELECTED_DIR)

# Historical plotting/reporting command; uses paths.best_model_path in CONFIG.
evaluate-legacy:
	$(PYTHON) -m src.evaluation.evaluate --config $(CONFIG)

api:
	$(PYTHON) -m uvicorn app.api.main:app --host $(API_HOST) --port $(API_PORT)

ui:
	$(PYTHON) -m streamlit run app/ui/streamlit_app.py
