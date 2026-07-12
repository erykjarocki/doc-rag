# PDF-RAG

Local RAG system for PDF documents — chat with your books, articles, or any PDF via AI.

Ask questions, get answers based **only on your PDF content**, with citations (book, chapter, page).

## How it works

```
PDF → text extraction → chunks → embeddings (local) → Qdrant (vector DB)
                                                          ↓
OpenCode (MCP) ← search_book_tool ← retriever ← similarity search
    ↓
LLM answers based on found fragments
```

- **100% local** — no data leaves your machine
- **OpenCode integration** — works as an MCP tool
- **Any PDF** — books, papers, manuals, your notes
- **Always cites sources** — book/chapter/page

## Quick start

### Prerequisites

- Python 3.10+
- Docker (for Qdrant)
- Git

### Setup

```bash
# 1. Clone
git clone git@github.com:erykjarocki/pdf-rag.git
cd pdf-rag

# 2. Setup (creates venv + installs everything)
make setup
source venv/bin/activate

# 3. Start Qdrant (Docker) — first time
make qdrant

# 4. Copy your PDFs
cp /path/to/your/files/*.pdf books/

# 5. Index everything
make ingest

# 6. Run MCP server (standalone test)
make mcp
```

### Day-to-day usage

```bash
# Start Qdrant (if stopped)
make start

# Index new PDFs or re-index
make ingest

# Re-index a specific book
make ingest ARGS="--reindex tom1"

# List indexed books
make ingest ARGS="--list"

# Run API server
make serve

# Run MCP server
make mcp

# Lint & format
make lint
make fmt
```

### All commands

```
make setup      # First-time setup (venv + install)
make install    # Reinstall package
make qdrant     # Start Qdrant (first time)
make start      # Start Qdrant (stopped container)
make stop       # Stop Qdrant
make ingest     # Index PDFs
make serve      # Run REST API
make mcp        # Run MCP server
make lint       # Lint with ruff
make fmt        # Format with ruff
make clean      # Remove __pycache__
```

## OpenCode integration

Add to your OpenCode config (`~/.config/opencode/opencode.json`):

```json
"pdf-rag": {
  "type": "local",
  "command": [
    "/path/to/pdf-rag/venv/bin/python",
    "/path/to/pdf-rag/src/mcp_server.py"
  ],
  "enabled": true
}
```

Then OpenCode automatically uses `search_book_tool` when you ask about your PDFs.

## Usage

### Index books
```bash
# Index all PDFs (each becomes its own collection)
make ingest

# Re-index a specific book
make ingest ARGS="--reindex tom1"

# Delete a book from the knowledge base
make ingest ARGS="--delete tom1"

# List indexed books
make ingest ARGS="--list"
```

### MCP tools

- `search_book_tool(question, book=None)` — search all books or filter by name
- `search_book_raw(question, book=None)` — returns JSON with scores
- `list_books_tool()` — list available books

## Project structure

```
pdf-rag/
├── books/              # Place your PDFs here
├── data/
│   ├── extracted/      # Raw text from PDFs
│   ├── chunks/         # Processed chunks
│   └── metadata/       # Index metadata
├── vector_db/qdrant/   # Qdrant storage
├── src/
│   ├── config.py       # Configuration
│   ├── ingest.py       # PDF → Qdrant pipeline
│   ├── embeddings.py   # Local embedding model
│   ├── retriever.py    # search_book() function
│   ├── mcp_server.py   # MCP server for OpenCode
│   ├── api.py          # REST API (optional)
│   └── qdrant_store.py # Qdrant client helpers
├── venv/
├── pyproject.toml      # Project config & dependencies
├── Makefile            # Quick commands
└── README.md
```

## Requirements

- Python 3.10+
- Docker (for Qdrant vector database)
- RAM: min 4 GB
- Disk: ~1.5 GB

## Changing the embedding model

The default model is `intfloat/multilingual-e5-small` (384 dim) — fast, local, handles Polish well.

To upgrade or switch:

```bash
# 1. Edit src/config.py:
#    EMBED_MODEL = "intfloat/multilingual-e5-large"
#    EMBED_DIM = 1024   # must match the model's dimension

# 2. Re-index (old collections are replaced automatically)
make ingest ARGS="--reindex investor-tom1"
```

**Model options** (full list on [huggingface.co/intfloat](https://huggingface.co/intfloat)):

| Model | Dimensions | Quality | Speed |
|---|---|---|---|
| `multilingual-e5-small` | 384 | good | fast |
| `multilingual-e5-base` | 768 | better | medium |
| `multilingual-e5-large` | 1024 | best | slow |

After changing the model you **must reindex** — old embeddings use a different dimension.

## Adding more documents

```bash
# Add a new PDF — just copy and index
cp ~/Downloads/new-book.pdf books/
make ingest
# → new collection created, existing ones untouched

# Re-index a specific book
make ingest ARGS="--reindex tom1"

# Remove a book
make ingest ARGS="--delete tom3"

# See what's indexed
make ingest ARGS="--list"
```
