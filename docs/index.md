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

```bash
make setup              # first-time setup
source venv/bin/activate
make qdrant             # start Qdrant (Docker)
make ingest             # index your PDFs
make mcp                # run MCP server
```

## Documentation

| Page | Description |
|------|-------------|
| [Architecture](architecture.md) | System overview and data flow |
| [Configuration](configuration.md) | All config options and how to change them |
| [API Reference](api-reference.md) | REST API and MCP tools |
| [Troubleshooting](troubleshooting.md) | Common issues and fixes |
