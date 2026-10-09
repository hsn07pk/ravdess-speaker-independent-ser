PY ?= python
PY_E2V ?= $(PY)
export PYTHONPATH := $(CURDIR)/src
export PYTHONWARNINGS := ignore::FutureWarning
EVAL_LOGS := logs/evaluation

.DEFAULT_GOAL := help
.PHONY: help install data features evaluate analysis sensitivity figures report test lint clean

help: ## Show this help
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-12s %s\n", $$1, $$2}'

install: ## Install the package with the pinned analysis stack and the dev tools
	$(PY) -m pip install -r requirements.txt -e ".[dev]"

data: ## Download RAVDESS speech from Zenodo into data/RAVDESS_speech
	bash scripts/download_ravdess.sh

features: ## Extract handcrafted features and frozen embeddings (needs the audio)
	PY=$(PY) PY_E2V=$(PY_E2V) bash scripts/extract_features.sh

evaluate: ## Run every LOSO evaluation from the features
	PY=$(PY) bash scripts/evaluate.sh

analysis: ## Summary tables, confidence intervals, paired tests and human comparison from the saved predictions
	$(PY) -m ser.analysis > $(EVAL_LOGS)/analysis.log

sensitivity: ## C-grid and SVM decision-rule sensitivity runs (needs the features)
	$(PY) -m ser.sensitivity > $(EVAL_LOGS)/sensitivity.log

figures: ## Figures of the report from the saved predictions
	$(PY) -m ser.figures > $(EVAL_LOGS)/figures.log

report: ## Build report/report.pdf
	cd report && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp report/main.pdf report/report.pdf

test: ## Run the unit tests
	$(PY) -m pytest -q

lint: ## Lint the code
	$(PY) -m ruff check .

clean: ## Remove the LaTeX build files
	cd report && latexmk -C main.tex
