# PDF-RAG Improvement Roadmap

## How to think about RAG

A RAG system is a **funnel** with three stages:

```
PDFs → CHUNKS (thousands) → TOP-8 vector results → RERANKED → LLM
         ↑                        ↑                    ↑
    Chunking quality         Retrieval quality      Precision
```

### Two Axes of RAG Quality

| Axis | What it means | Failure symptom |
|---|---|---|
| **Recall** | Did we find ALL relevant passages? | "The book says it, but RAG didn't find it" |
| **Precision** | Are ALL returned passages relevant? | "The answer is mixed with irrelevant junk" |

These **trade off** — improving recall often reduces precision. Reranking recovers precision after boosting recall.

### Three Levers (cheapest → most expensive)

1. **Chunking** — how you slice the book
2. **Retrieval** — how you find relevant chunks
3. **Ranking** — how you order results for the LLM

### The MCP Agent Dimension

This RAG system is used by an **LLM agent** (opencode), not directly by users:

- The agent **decides when to call** based on tool descriptions
- The agent **parses results** to answer — formatting matters
- The agent **may call multiple tools** — speed and concurrency matter
- The agent has **limited context** — don't overwhelm it

Tool description quality + speed/reliability often matter more than marginal retrieval gains.

### Iteration Process

```
1. Identify the failure mode
2. Pick the cheapest lever
3. Make one change
4. Test (same question before and after)
5. Repeat
```

---

## Improvement List

### 1. Fix broken venv pip
- **Effort:** trivial
- **Problem:** `venv/bin/pip3` has old shebang path
- **Fix:** Recreate venv or edit shebangs
- **Why:** Can't install/upgrade dependencies without it

### 2. Add pyproject.toml
- **Effort:** 10 min
- **Problem:** No `pyproject.toml`, bare `requirements.txt`, `sys.path` hacks everywhere
- **Fix:** Create `pyproject.toml`, remove `sys.path.insert` from scripts
- **Why:** Enables `pip install -e .`, locks versions, removes hacks

### 3. Cache chunks to disk
- **Effort:** 15 min
- **Problem:** Re-indexing re-chunks everything even when only embedding model changes
- **Fix:** Save chunks to JSON after `process_book()`, load cached chunks when available
- **Why:** Cuts re-index time from minutes to seconds when only re-embedding

### 4. Async MCP tools
- **Effort:** 10 min
- **Problem:** Tools are synchronous, block the MCP event loop
- **Fix:** Make tools `async def`, wrap sync calls in `run_in_executor` or use `AsyncQdrantClient`
- **Why:** Enables concurrent tool calls, faster response

### 5. Better MCP tool descriptions
- **Effort:** 5 min
- **Problem:** Generic descriptions, agent may not use tools optimally
- **Fix:** More specific descriptions with arg docs and usage context
- **Why:** Better agent behavior = better user experience

### 6. Configurable top_k per request
- **Effort:** 5 min
- **Problem:** `TOP_K = 8` is hardcoded in config.py, not exposed in MCP tools
- **Fix:** Add `top_k` parameter to MCP tools and API endpoint
- **Why:** Broad queries need more results, specific queries need fewer

### 7. Add logging / observability
- **Effort:** 15 min
- **Problem:** No logging — zero insight into failures, slow queries, or bad results
- **Fix:** Add `logging` to retriever, qdrant_store, ingest; configure in mcp_server.py
- **Why:** Debuggability — know why empty results or slow searches happen

### 8. Clean up Qdrant config confusion
- **Effort:** 5 min
- **Problem:** `config.py` has unused `QDRANT_PATH` and uses `QDRANT_HOST:PORT` (Docker)
- **Fix:** Remove dead `QDRANT_PATH`, or switch to embedded mode
- **Why:** Dead code is misleading

### 9. Hybrid search (BM25 + vector)
- **Effort:** 1-2 hrs
- **Problem:** Pure vector search misses exact keyword matches
- **Fix:** Add sparse vectors (BM25) to Qdrant collections, use hybrid queries with RRF fusion
- **Why:** Biggest retrieval quality improvement — catches both semantic and exact matches

### 10. Add reranking
- **Effort:** 1 hr
- **Problem:** Top-8 by vector similarity ≠ top-8 by relevance
- **Fix:** Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores candidates
- **Why:** +10-20% retrieval accuracy, eliminates false positives

### 11. Better tokenizer-based chunking
- **Effort:** 30 min
- **Problem:** Char-count heuristic (~4 chars/token) is imprecise, creates uneven chunks
- **Fix:** Use the actual tokenizer for exact 400-token chunks with precise overlap
- **Why:** Consistent chunks = predictable retrieval quality

### 12. Add tests
- **Effort:** 1-2 hrs
- **Problem:** Zero tests — no safety net for changes
- **Fix:** Unit tests for chunking, retriever, embedding; integration test with test PDF
- **Why:** Catch regressions before they reach production

### 13. Async ingestion pipeline
- **Effort:** 1 hr
- **Problem:** PDF processing runs sequentially, slow for large books
- **Fix:** `asyncio.gather` for parallel PDF processing, async Qdrant upserts
- **Why:** ~3x faster indexing for multiple books

### 14. Query expansion
- **Effort:** 1 hr
- **Problem:** Single query may not match varied phrasing in the book
- **Fix:** Synonym expansion or LLM-generated query variants before search
- **Why:** Better recall — finds passages even when wording differs

### 15. Dockerfile + docker-compose
- **Effort:** 1 hr
- **Problem:** No reproducible deployment, setup depends on local env
- **Fix:** Dockerfile for pdf-rag, docker-compose with Qdrant service
- **Why:** One-command setup, CI/CD ready

### 16. Upgrade embedding model
- **Effort:** 15 min + re-index
- **Problem:** `multilingual-e5-small` (384d) is the smallest E5 variant
- **Fix:** Switch to `multilingual-e5-large` (1024d) or `BAAI/bge-m3`
- **Why:** Best semantic quality, but requires re-indexing

---

## Typical Failure Modes Quick Reference

| Symptom | Likely cause | Cheapest fix |
|---|---|---|
| "Didn't find the passage I know exists" | Low recall | Query expansion or hybrid search (#9, #14) |
| "Found too much irrelevant stuff" | Low precision | Reranking (#10) |
| "Answer is cut off / partial" | Bad chunk boundaries | Semantic chunking by chapter (#11) |
| "Can't answer simple facts" | Chunks too large | Smaller chunks (200-300 tokens) |
| "Agent doesn't call the tool" | Bad description | Better tool descriptions (#5) |
| "Slow responses" | Sync blocking | Async tools (#4) |
| "Empty results, no error" | No observability | Add logging (#7) |
| "Re-indexing takes forever" | No chunk cache | Chunk caching (#3) |

## Status

<!-- Update as items are completed -->
- [ ] #1 Fix pip
- [ ] #2 pyproject.toml
- [ ] #3 Chunk caching
- [ ] #4 Async MCP
- [ ] #5 Better descriptions
- [ ] #6 Configurable top_k
- [ ] #7 Logging
- [ ] #8 Clean config
- [ ] #9 Hybrid search
- [ ] #10 Reranking
- [ ] #11 Better chunking
- [ ] #12 Tests
- [ ] #13 Async ingestion
- [ ] #14 Query expansion
- [ ] #15 Docker
- [ ] #16 Better model
