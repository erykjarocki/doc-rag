# Future Improvements

## Search Quality

### Hybrid Search (BM25 + Semantic)
- Add BM25 keyword search alongside vector similarity
- Use Qdrant's built-in sparse vector support for BM25
- Combine scores with configurable weighting (e.g. 0.7 semantic + 0.3 BM25)
- Better for exact term matches (names, codes, technical terms)

### Reranking
- Add a cross-encoder reranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- rerank top 20 results down to top 8
- Significant quality improvement at cost of latency (~50ms)

### Query Expansion
- Generate sub-queries from the original question
- Search with multiple queries, merge results
- Helps with vague or multi-faceted questions

### Metadata Filtering
- Filter by chapter, page range, or custom tags
- Store additional metadata during ingestion (author, date, tags)
- API parameter for filtered search

## Web UI

### Minimal Search Interface
- Single page with search input and results display
- Shows source citations with clickable references
- Collection browser sidebar
- Built with vanilla HTML/JS or htmx (no build step)

### Document Management
- Upload documents via drag-and-drop
- View indexed collections with chunk counts
- Delete documents individually
- Re-index with progress indication

### Tech Stack
- FastAPI serves the UI as static files
- No separate frontend build needed
- WebSocket for ingestion progress updates

## Implementation Priority

1. **Hybrid Search** — biggest quality improvement
2. **Reranking** — easy win, significant quality boost
3. **Web UI (search)** — makes system accessible to non-technical users
4. **Web UI (management)** — complete the CRUD cycle
5. **Query Expansion** — incremental quality improvement
6. **Metadata Filtering** — power user feature
