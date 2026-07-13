.PHONY: setup install ingest serve mcp qdrant start stop clean lint fmt docs docs-serve test test-unit test-integration

PY = venv/bin/python
PIP = venv/bin/pip
MKDOCS = venv/bin/mkdocs
RUFF = venv/bin/ruff
PYTEST = venv/bin/pytest

# Setup (first time)
setup:
	python3 -m venv venv
	$(PIP) install -e ".[dev]"
	@echo "\n✅ Done. Run commands via 'make <cmd>'"

install:
	$(PIP) install -e ".[dev]"

# Qdrant
qdrant:
	docker run -d --name qdrant -p 6333:6333 \
		-v $(PWD)/vector_db/qdrant:/qdrant/storage \
		qdrant/qdrant

start:
	docker start qdrant

stop:
	docker stop qdrant

# PDF pipeline
ingest:
	$(PY) src/ingest.py $(ARGS)

serve:
	$(PY) src/api.py

mcp:
	$(PY) src/mcp_server.py

# Dev tools
lint:
	$(RUFF) check src/ $(ARGS)

fmt:
	$(RUFF) format src/ $(ARGS)

test:
	$(PYTEST) tests/ -v

test-unit:
	$(PYTEST) tests/unit/ -v -m unit

test-integration:
	$(PYTEST) tests/integration/ -v -m integration

test-eval:
	$(PYTEST) tests/eval/ -v -m eval

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Documentation
docs:
	$(MKDOCS) build

docs-serve:
	$(MKDOCS) serve
