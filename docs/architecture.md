# Architecture

## System Overview

PDF-RAG is a local Retrieval-Augmented Generation system that lets you query PDF documents via semantic search. It has two pipelines: **ingestion** (PDF → vector DB) and **retrieval** (query → relevant chunks).

## Data Flow

### Ingestion Pipeline

```
books/*.pdf
    │
    ▼
┌─────────────────┐
│   extract_pdf()  │  PyMuPDF reads each page
│   (ingest.py)    │
└────────┬────────┘
         │  list[dict] with page_num + text
         ▼
┌─────────────────┐
│  chunk_text()    │  Tokenizer-aware splitting (~384 tokens)
│  (ingest.py)     │  with 50-token overlap + page tracking
└────────┬────────┘
         │  list[dict] with text + start_page + end_page
         ▼
┌─────────────────────────────────────┐
│  ChapterDetector                    │  3-layer fallback:
│  (chapter_detection.py)             │  1. PDF TOC/bookmarks (best)
│                                     │  2. Font-size analysis
│                                     │  3. Regex pattern matching
└────────┬────────────────────────────┘
         │  chunks annotated with chapter names
         ▼
┌─────────────────┐
│    embed()       │  Batch encode with multilingual-e5-small
│  (embeddings.py) │  (passage: prefix for E5 models)
└────────┬────────┘
         │  384-dim normalized vectors
         ▼
┌─────────────────┐
│  qdrant.upsert() │  Store vectors + metadata in Qdrant
│ (qdrant_store.py)│  Batched in groups of 500
└────────┬────────┘
         │
         ▼
   Qdrant (Docker, localhost:6333)
   Persistent storage in vector_db/qdrant/
```

### Retrieval Pipeline

```
User question
    │
    ▼
┌────────────────────┐
│  embed_query()      │  Encode query (query: prefix for E5)
│  (embeddings.py)    │
└────────┬───────────┘
         │  384-dim query vector
         ▼
┌────────────────────┐
│  client.query_      │  Cosine similarity search
│  points()           │  Per collection or all collections
│  (qdrant_store.py)  │
└────────┬───────────┘
         │  top-8 scored results
         ▼
┌────────────────────┐
│  format_fragments_  │  Numbered text blocks with
│  for_prompt()       │  Polish source citations
│  (retriever.py)     │
└────────┬───────────┘
         │  Formatted string
         ▼
    LLM agent generates answer
```

## Component Responsibilities

| Module | Responsibility | Key Functions |
|--------|---------------|---------------|
| `config.py` | Central configuration | Paths, model name, chunk size, collection naming |
| `embeddings.py` | Text ↔ vector conversion | `embed()`, `embed_query()`, `get_model()` |
| `qdrant_store.py` | Vector DB connection | `ensure_collection()`, `list_collections()` |
| `chapter_detection.py` | Chapter/section detection | `ChapterDetector`, `_build_toc_map()`, `_build_font_map()` |
| `ingest.py` | PDF processing pipeline | `extract_pdf()`, `chunk_text()`, `index_book()` |
| `retriever.py` | Search and formatting | `search_book()`, `format_fragments_for_prompt()` |
| `mcp_server.py` | OpenCode integration | `search_book_tool()`, `list_books_tool()` |
| `api.py` | REST API (optional) | `/query`, `/books`, `/health` |

## Design Decisions

### Why per-book collections?
Each PDF gets its own Qdrant collection. This enables:
- Searching a specific book without noise from others
- Easy deletion/re-indexing of individual books
- Collection-level statistics

### Why E5 with prefixes?
E5 models are trained with `passage:` and `query:` prefixes to distinguish between indexed text and search queries. This improves retrieval quality significantly compared to unprefixed embeddings.

### Why tokenizer-aware chunking?
Character-count heuristics (e.g., "4 chars per token") are inaccurate for multilingual text. Using the actual tokenizer produces consistent chunk sizes regardless of language.

### Why batch upserts?
Qdrant recommends upserting in batches of ≤500 points for optimal performance. Large single upserts can cause timeouts or memory issues with large books.

### Why local-only?
No data leaves the machine. The embedding model runs locally via `sentence-transformers`. Qdrant runs in Docker with persistent local storage. This is critical for privacy-sensitive documents.

### Why three-layer chapter detection?
Production RAG systems use structural PDF metadata, not regex on extracted text. The `ChapterDetector` tries strategies in order of reliability:

1. **PDF TOC/bookmarks** (`doc.get_toc()`) — fastest and most reliable when the PDF includes a bookmark structure. Builds a page→breadcrumb mapping and fills gaps between entries.

2. **Font-size analysis** (`page.get_text("dict")`) — extracts per-span font metadata, computes the mean font size (body text dominates the distribution), and classifies headings using calibrated thresholds (`mean + 4pt` for headings, `mean + 2pt` for subheadings). Also checks bold + ALL-CAPS as additional signals.

3. **Regex fallback** — language-agnostic patterns covering English, Polish, numbered headings, and legal formats. Last resort when no structural data exists.

Each strategy is lazy-evaluated: only the first successful strategy is used. This approach works for any PDF regardless of language or formatting, unlike the previous regex-only detection which was limited to Polish and English chapter patterns.
