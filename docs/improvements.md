# Improvement Roadmap

| # | Improvement | Status | Effort | Impact | Notes |
|---|------------|--------|--------|--------|-------|
| 1 | Fix broken venv pip | ✅ | trivial | blocker | Recreated venv |
| 2 | Add pyproject.toml | ✅ | 10 min | high | Enables `pip install -e .`, entry points, locked deps |
| 3 | Unit & integration tests | ✅ | 2 hrs | high | 124 unit tests, integration tests, eval suite |
| 4 | Structured logging | ✅ | 15 min | medium | `src/log.py`, text + JSON formatters, request tracing |
| 5 | Clean up Qdrant config | ✅ | 5 min | low | Removed dead `QDRANT_PATH`, env-var-driven config |
| 6 | Better MCP tool descriptions | ✅ | 5 min | medium | Detailed arg docs, usage context, agent guidance |
| 7 | Configurable `top_k` per request | ⬜ | 5 min | low | Not exposed in MCP tools or API endpoint |
| 8 | Cache chunks to disk | ⬜ | 15 min | low | Saves re-chunking when only re-embedding |
| 9 | Tokenizer-based chunking | ✅ | 30 min | high | Exact token counts via model tokenizer, 384-token chunks |
| 10 | Universal chapter detection | ✅ | 2 hrs | high | 3-layer fallback: TOC → font-size → regex |
| 11 | Async MCP tools | ⬜ | 10 min | low | Currently sync, blocks MCP event loop |
| 12 | Async ingestion pipeline | ⬜ | 1 hr | medium | Parallel PDF processing, async Qdrant upserts |
| 13 | Hybrid search (BM25 + vector) | ⬜ | 1–2 hrs | **high** | Biggest retrieval improvement — catches exact + semantic matches |
| 14 | Cross-encoder reranking | ✅ | 1 hr | high | `ms-marco-MiniLM-L-6-v2`, re-scores top-20 → top-8 |
| 15 | Query expansion / rewriting | ⬜ | 1 hr | medium | Synonym expansion or LLM query variants |
| 16 | Dockerfile + docker-compose | ✅ | 1 hr | medium | Qdrant + API services, one-command setup |
| 17 | Upgrade embedding model | ✅ | 15 min | high | e5-small (384d) → e5-base (768d), better retrieval quality |
| 18 | Dimension mismatch protection | ✅ | 15 min | medium | Hard error on search, warning on ingest, no hardcoded dims |

## Priority order (for next improvements)

| Priority | Item | Why |
|----------|------|-----|
| 1 | Hybrid search BM25+vector (#13) | Biggest retrieval quality gain, catches keyword matches vector search misses |
| 2 | Query expansion/rewriting (#15) | Improves recall for varied phrasing |
| 3 | Configurable `top_k` per request (#7) | Quick win, broad queries need more results |
| 4 | Cache chunks to disk (#8) | Faster iteration during development |
| 5 | Async ingestion pipeline (#12) | Faster indexing for multi-document workflows |
| 6 | Async MCP tools (#11) | Enables concurrent tool calls |
