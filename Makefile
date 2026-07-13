# Agentic Research & Report Assistant — common tasks (keyless by default).
PYTHON ?= python3
export PYTHONPATH := src

.PHONY: help install run api ui eval eval-compare eval-compare-backends optimize dspy-install test lint fmt docker clean

help:
	@echo "targets: install run api ui eval eval-compare test lint fmt docker clean"
	@echo "dspy:    dspy-install optimize eval-compare-backends  (optional DSPy track)"
	@echo 'usage:   make run Q="What are the trade-offs of RAG retrieval methods?"'

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	$(PYTHON) -m agent.runner "$(Q)"

api:
	$(PYTHON) -m uvicorn agent.api:app --host 0.0.0.0 --port 8000 --reload

ui:
	$(PYTHON) -m streamlit run ui/streamlit_app.py

eval:
	$(PYTHON) -m eval.run_eval

eval-compare:
	$(PYTHON) -m eval.run_eval --compare

eval-compare-backends:
	$(PYTHON) -m eval.run_eval --compare-backends

optimize:  ## DSPy: compile/optimize the program (needs OPENAI_API_KEY)
	$(PYTHON) -m agent.optimize

dspy-install:  ## install the optional DSPy dependency
	$(PYTHON) -m pip install "dspy-ai>=3.0,<4"

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff format .

docker:
	docker compose up --build api

clean:
	rm -rf runs eval/results .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +
