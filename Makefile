.PHONY: setup install serve mcp qdrant start stop clean lint fmt docs docs-serve test test-unit test-integration eval-sweep eval-sweep-parallel docker-up docker-down

PY = venv/bin/python
PIP = venv/bin/pip
MKDOCS = venv/bin/mkdocs
RUFF = venv/bin/ruff
PYTEST = venv/bin/pytest

# Setup (first time) — one command to get started
setup:
	python3 -m venv venv
	$(PIP) install -e ".[dev]"
	@$(PY) -m src.config_cli init 2>/dev/null || true
	@echo ""
	@echo "=== Starting Qdrant ==="
	@if docker ps -a --format '{{.Names}}' | grep -q '^qdrant$$'; then \
		docker start qdrant 2>/dev/null && echo "Qdrant started." || echo "Qdrant already running."; \
	else \
		docker run -d --name qdrant -p 6333:6333 \
			-v $(PWD)/vector_db/qdrant:/qdrant/storage \
			qdrant/qdrant && echo "Qdrant started." || echo "Docker not available — start Qdrant manually."; \
	fi
	@echo ""
	@echo "=== Config ==="
	@echo "Config file: ~/.config/doc-rag/config.json"
	@echo "Edit this file to change model, ports, chunk size, etc."
	@echo ""
	@echo "=== OpenCode Integration ==="
	@echo "Add this to ~/.config/opencode/opencode.json:"
	@echo ""
	@echo '  "doc-rag": {'
	@echo '    "type": "local",'
	@echo '    "command": ["$(PWD)/venv/bin/python", "-m", "src.mcp_server"],'
	@echo '    "cwd": "$(PWD)",'
	@echo '    "enabled": true'
	@echo '  }'
	@echo ""
	@echo "Done! Restart OpenCode, then ask it to ingest documents."

install:
	$(PIP) install -e ".[dev]"

# Docker Compose
docker-up:
	docker compose up -d --build
	@echo "API running at http://localhost:$${API_PORT:-8000}"
	@echo "Qdrant running at http://localhost:$${QDRANT_PORT:-6333}"

docker-down:
	docker compose down

# Qdrant
qdrant:
	docker run -d --name qdrant -p 6333:6333 \
		-v $(PWD)/vector_db/qdrant:/qdrant/storage \
		qdrant/qdrant

start:
	@docker start qdrant 2>/dev/null || echo "Qdrant not found. Run 'make qdrant' first."

stop:
	docker stop qdrant

# Servers
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
	$(PYTEST) tests/eval/ -v -m "eval or rerank"

eval-sweep:
	$(PY) tests/eval/run_sweep.py

eval-sweep-parallel:
	$(PY) tests/eval/run_sweep.py --parallel

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true

# Documentation
docs:
	$(MKDOCS) build

docs-serve:
	$(MKDOCS) serve
