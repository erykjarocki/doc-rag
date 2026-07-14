# Configuration

DOC-RAG uses a 3-tier configuration system. Settings are resolved in this order (highest priority first):

1. **Environment variables** — override everything
2. **Config file** — `~/.config/doc-rag/config.json`
3. **Code defaults** — built into `src/config.py`

### Creating a config file

```bash
# Generate default config file
make setup
# or:
python -m src.config_cli init
```

This creates `~/.config/doc-rag/config.json`. Edit it to customize settings without touching source code.

### Environment variables

Every setting can be overridden via environment variables. This is useful for CI, Docker, or quick testing:

```bash
EMBED_MODEL=intfloat/multilingual-e5-large EMBED_DIM=1024 python src/ingest.py --reindex
```

## Configuration Options

### Paths

| Variable | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `BASE_DIR` | Project root | — | Auto-detected from file location |
| `EXTRACTED_DIR` | `data/extracted/` | — | Where raw extracted text is saved (.txt) |
| `CHUNKS_FILE` | `data/chunks/chunks.json` | — | Reserved for cached chunks (not yet used) |
| `METADATA_FILE` | `data/metadata/metadata.json` | — | Reserved for index metadata (not yet used) |
| `QDRANT_PATH` | `vector_db/qdrant` | — | Qdrant storage path (unused, Docker mode active) |

### Embedding Model

| Variable | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `EMBED_MODEL` | `intfloat/multilingual-e5-base` | `EMBED_MODEL` | Sentence-transformers model name |
| `EMBED_DIM` | `768` | `EMBED_DIM` | Vector dimensions (must match model) |

### Qdrant

| Variable | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `QDRANT_HOST` | `localhost` | `QDRANT_HOST` | Qdrant Docker host |
| `QDRANT_PORT` | `6333` | `QDRANT_PORT` | Qdrant Docker port |

### Chunking

| Variable | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `CHUNK_SIZE` | `384` | `CHUNK_SIZE` | Target tokens per chunk |
| `CHUNK_OVERLAP` | `50` | `CHUNK_OVERLAP` | Overlap tokens between adjacent chunks |

### Retrieval

| Variable | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `TOP_K` | `8` | `TOP_K` | Default number of results returned per query |

### Reranking

| Variable | Default | Env Var | Description |
|----------|---------|---------|-------------|
| `RERANK_ENABLED` | `true` | `RERANK_ENABLED` | Enable cross-encoder re-ranking stage |
| `RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `RERANK_MODEL` | HuggingFace CrossEncoder model |
| `RERANK_TOP_N` | `20` | `RERANK_TOP_N` | Candidates to retrieve before re-ranking |

---

## Supported Formats

The system supports 40+ file formats via adapters:

| Format | Extensions | Sections detected |
|--------|-----------|-------------------|
| PDF | `.pdf` | Chapter detection (TOC → font analysis → regex) |
| Markdown | `.md`, `.markdown` | `#` headings |
| Source code | `.py`, `.js`, `.ts`, `.rs`, `.go`, `.java`, etc. | Functions, classes |
| Plain text | `.txt`, `.log`, `.csv`, `.json`, `.xml`, etc. | None (single section) |

See `src/adapters.py` for the complete list of supported extensions.

---

## Changing the Embedding Model

Use environment variables or edit `~/.config/doc-rag/config.json`:

```bash
# Via environment variables (temporary)
EMBED_MODEL=intfloat/multilingual-e5-large EMBED_DIM=1024 python src/ingest.py --reindex

# Via config file (permanent)
# Edit ~/.config/doc-rag/config.json:
# {
#   "embedding": {
#     "model": "intfloat/multilingual-e5-large",
#     "dimension": 1024
#   }
# }
```

| Model | Dimensions | Quality | Speed | RAM |
|---|---|---|---|---|
| `multilingual-e5-small` | 384 | good | fast | ~1 GB |
| `multilingual-e5-base` | 768 | better | medium | ~2 GB |
| `multilingual-e5-large` | 1024 | best | slow | ~4 GB |

Then **re-index all documents** (old collections have wrong dimensions):

```bash
python src/ingest.py /path/to/document.pdf --reindex
python src/ingest.py --folder /path/to/documents/ --reindex
```

The system detects dimension mismatches automatically and will error before querying a collection indexed with the wrong model.

---

## Qdrant Setup

### First time

```bash
docker run -d --name qdrant -p 6333:6333 \
  -v $(pwd)/vector_db/qdrant:/qdrant/storage \
  qdrant/qdrant
```

Or use `make setup` which handles this automatically.

### Subsequent starts

```bash
make start
```

### Check status

```bash
curl http://localhost:6333/health
```

---

## Changing Chunk Size

Larger chunks = more context per result, but fewer total chunks. Smaller chunks = more precise retrieval, but may split related content.

Use environment variables or edit `~/.config/doc-rag/config.json`:

```bash
# Via environment variables (temporary)
CHUNK_SIZE=200 CHUNK_OVERLAP=30 python src/ingest.py --reindex

# Via config file (permanent)
# {
#   "chunking": {
#     "size": 200,
#     "overlap": 30
#   }
# }
```

After changing, re-index affected documents.
