.PHONY: setup install ingest serve mcp qdrant start stop clean lint fmt

# Setup (first time)
setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -e ".[dev]"
	@echo "\n✅ Done. Use 'source venv/bin/activate' or run commands via 'make <cmd>'"

install:
	pip install -e ".[dev]"

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
	python src/ingest.py $(ARGS)

serve:
	python src/api.py

mcp:
	python src/mcp_server.py

# Dev tools
lint:
	ruff check src/

fmt:
	ruff format src/

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
