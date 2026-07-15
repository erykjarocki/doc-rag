# Architecture

## System Overview

DOC-RAG is a local Retrieval-Augmented Generation system that lets you query documents via semantic search. It supports PDFs, Markdown, source code, and plain text files. It has two pipelines: **ingestion** (document → vector DB) and **retrieval** (query → relevant chunks).

## Data Flow

### Ingestion Pipeline

```
/path/to/document.{pdf,md,py,txt,...}
    │
    ▼
┌─────────────────────┐
│  get_adapter(path)  │  Dispatch by file extension
│  (adapters.py)      │
└────────┬────────────┘
         │  PDFAdapter / MarkdownAdapter / CodeAdapter / PlainTextAdapter
         ▼
┌─────────────────────┐
│  adapter.extract()   │  Format-specific text extraction
│  (adapters.py)       │  Returns Document with sections
└────────┬────────────┘
         │  Document(full_text, page_boundaries, sections)
         ▼
┌─────────────────────┐
│   chunk_text()       │  Tokenizer-aware splitting (~384 tokens)
│   (chunking.py)      │  with 50-token overlap + page tracking
└────────┬────────────┘
         │  list[dict] with text + start_page + end_page
         ▼
┌─────────────────────────────────────┐
│    embed()       │  Batch encode with multilingual-e5-base
│  (embeddings.py) │  (passage: prefix for E5 models)
└────────┬────────────────────────────┘
         │  768-dim normalized vectors
         ▼
┌─────────────────────┐
│  qdrant.upsert()    │  Store vectors + metadata in Qdrant
│ (qdrant_store.py)   │  Batched in groups of 500
└────────┬────────────┘
         │
         ▼
   Qdrant (Docker, localhost:6333)
   Persistent storage in vector_db/qdrant/
```

Documents are ingested via:
- **MCP tools**: `ingest_document()` or `ingest_folder()` from OpenCode
- **CLI**: `python src/ingest.py <file>` or `python src/ingest.py --folder <dir>`

### Retrieval Pipeline

```
User question
    │
    ▼
┌────────────────────┐
│  embed_query()      │  Encode query (query: prefix for E5)
│  (embeddings.py)    │
└────────┬───────────┘
         │  768-dim query vector
         ▼
┌─────────────────────────────────────────┐
│  client.query_points()                   │  Cosine similarity search
│  (qdrant_store.py)                       │  Per collection or all collections
└────────┬────────────────────────────────┘
         │  top-20 candidates (when reranking enabled)
         ▼
┌─────────────────────────────────────────┐
│  rerank()                               │  Cross-encoder rescores each
│  (reranker.py)                          │  (query, document) pair jointly
└────────┬────────────────────────────────┘
         │  top-8 results with rerank scores
         ▼
┌─────────────────────────────────────────┐
│  format_fragments_for_prompt()          │  Numbered text blocks with
│  (retriever.py)                         │  Polish source citations
└────────┬────────────────────────────────┘
         │  Formatted string
         ▼
    LLM agent generates answer
```

## Bi-Encoder vs Cross-Encoder: Two-Stage Retrieval

DOC-RAG uses a **two-stage retrieval** architecture for optimal speed and accuracy:

### Stage 1: Bi-Encoder (Fast Retrieval)

- **Model:** `intfloat/multilingual-e5-base` (768 dimensions)
- **How it works:** Encodes query and documents *separately* into vectors, then uses cosine similarity
- **Speed:** Very fast — can search millions of documents in milliseconds
- **Accuracy:** Good but imprecise — sees query and document independently
- **Purpose:** Narrow down millions of chunks to top-20 candidates

### Stage 2: Cross-Encoder (Precise Re-ranking)

- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **How it works:** Processes query and document *jointly* through a transformer, outputting a relevance score
- **Speed:** Slower — must process each (query, document) pair individually
- **Accuracy:** Much higher — sees both query and document simultaneously
- **Purpose:** Rescore top-20 candidates to find the true top-8

### Why Two Stages?

| Approach | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| Bi-encoder only | ⚡ Fast | 🎯 Good | Large-scale initial filtering |
| Cross-encoder only | 🐌 Slow | 🎯🎯 Excellent | Small candidate sets |
| **Two-stage (our approach)** | ⚡ Fast | 🎯🎯 Excellent | **Production RAG** |

The bi-encoder acts as a fast filter, reducing thousands of candidates to a manageable set. The cross-encoder then applies precise judgment to find the most relevant results.

### Configuration

Re-ranking is enabled by default. See [Configuration](configuration.md) for how to toggle it, change the model, or adjust the candidate count.

## Component Responsibilities

| Module | Responsibility | Key Functions |
|--------|---------------|---------------|
| `config.py` | Central configuration | Paths, model name, chunk size, collection naming |
| `adapters.py` | Format-specific extraction | `get_adapter()`, `PDFAdapter`, `MarkdownAdapter`, `CodeAdapter`, `PlainTextAdapter` |
| `embeddings.py` | Text ↔ vector conversion | `embed()`, `embed_query()`, `get_model()` |
| `reranker.py` | Cross-encoder re-ranking | `rerank()`, `rerank_with_analysis()`, `get_reranker()` |
| `qdrant_store.py` | Vector DB connection | `ensure_collection()`, `list_collections()` |
| `chapter_detection.py` | Chapter/section detection (PDF) | `ChapterDetector`, `_build_toc_map()`, `_build_font_map()` |
| `chunking.py` | Token-aware text splitting | `chunk_text()` |
| `ingest.py` | Document processing pipeline | `process_document()`, `index_document()`, `ingest_folder()` |
| `retriever.py` | Search and formatting | `search_book()`, `format_fragments_for_prompt()` |
| `mcp_server.py` | MCP server for AI assistants | `search_document_tool()`, `list_documents_tool()`, `ingest_document()`, `ingest_folder()` |

## Adapter Pattern

Each document format has its own adapter that handles extraction:

- **PDFAdapter**: Wraps PyMuPDF extraction + ChapterDetector for PDF-specific features (TOC, font analysis, regex fallback)
- **MarkdownAdapter**: Parses `#` headings as section boundaries, tracks line numbers
- **CodeAdapter**: Regex-based detection of functions, classes, and methods for 20+ programming languages
- **PlainTextAdapter**: Treats the entire file as a single section with one virtual page

All adapters return a `Document` object with the same interface, so the rest of the pipeline (chunking, embedding, storage) works unchanged.

## Design Decisions

### Why per-document collections?
Each document gets its own Qdrant collection. This enables:
- Searching a specific document without noise from others
- Easy deletion/re-indexing of individual documents
- Collection-level statistics

### Why E5 with prefixes?
E5 models are trained with `passage:` and `query:` prefixes to distinguish between indexed text and search queries. This improves retrieval quality significantly compared to unprefixed embeddings.

### Why tokenizer-aware chunking?
Character-count heuristics (e.g., "4 chars per token") are inaccurate for multilingual text. Using the actual tokenizer produces consistent chunk sizes regardless of language.

### Why batch upserts?
Qdrant recommends upserting in batches of ≤500 points for optimal performance. Large single upserts can cause timeouts or memory issues with large documents.

### Why local-only?
No data leaves the machine. The embedding model runs locally via `sentence-transformers`. Qdrant runs in Docker with persistent local storage. This is critical for privacy-sensitive documents.

### Why three-layer chapter detection?
Production RAG systems use structural PDF metadata, not regex on extracted text. The `ChapterDetector` tries strategies in order of reliability:

1. **PDF TOC/bookmarks** (`doc.get_toc()`) — fastest and most reliable when the PDF includes a bookmark structure. Builds a page→breadcrumb mapping and fills gaps between entries.

2. **Font-size analysis** (`page.get_text("dict")`) — extracts per-span font metadata, computes the mean font size (body text dominates the distribution), and classifies headings using calibrated thresholds (`mean + 4pt` for headings, `mean + 2pt` for subheadings). Also checks bold + ALL-CAPS as additional signals.

3. **Regex fallback** — language-agnostic patterns covering English, Polish, numbered headings, and legal formats. Last resort when no structural data exists.

Each strategy is lazy-evaluated: only the first successful strategy is used. This approach works for any PDF regardless of language or formatting.

### Why adapter pattern for formats?
Different document types need fundamentally different extraction logic. PDFs require PyMuPDF with font analysis; Markdown needs heading parsing; code needs function/class detection. The adapter pattern encapsulates this complexity while providing a uniform `Document` interface, so adding a new format requires only a new adapter class.

### Why no staging directory?
Documents are ingested directly from their original location via MCP tools or CLI. No intermediate folder to manage — works with any file path on disk.
