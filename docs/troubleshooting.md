# Troubleshooting

## Common Issues

### Qdrant connection refused

**Error:** `ConnectionError` or `Connection refused`

**Cause:** Qdrant Docker container isn't running.

**Fix:**
```bash
# Check if container exists
docker ps -a | grep qdrant

# Start it
docker start qdrant

# Or create fresh
docker run -d --name qdrant -p 6333:6333 \
  -v $(pwd)/vector_db/qdrant:/qdrant/storage \
  qdrant/qdrant
```

### Empty search results

**Symptom:** `search_book_tool` returns "No relevant fragments found"

**Possible causes:**
1. Book not indexed — run `python src/ingest.py --list` to check
2. Query doesn't match content — try broader terms
3. Wrong book name — use `list_books_tool()` to see exact names

**Fix:**
```bash
python src/ingest.py --list
# If missing, index it:
python src/ingest.py
```

### Import errors / ModuleNotFoundError

**Error:** `ModuleNotFoundError: No module named 'src'`

**Cause:** Running scripts from wrong directory or missing sys.path hack.

**Fix:**
```bash
# Always run from project root
cd /path/to/doc-rag
python src/ingest.py
```

### Slow ingestion

**Symptom:** Indexing takes very long

**Possible causes:**
1. Large PDF — normal for 500+ page books
2. First run downloads embedding model (~1.1GB for e5-base)
3. Re-chunking on every run (no chunk caching yet)

**Tip:** Check `data/extracted/` — if .txt exists, extraction was already done.

### Wrong chunk dimensions

**Error:** Qdrant rejects vectors with dimension mismatch

**Cause:** Changed `EMBED_DIM` without re-indexing.

**Fix:** Delete and re-index all collections:
```bash
python src/ingest.py --delete investor-tom1
python src/ingest.py --delete investor-tom2
python src/ingest.py  # re-index all
```

### MCP server won't start

**Symptom:** OpenCode can't connect to doc-rag tools

**Check:**
1. Virtual environment is activated
2. `fastmcp` is installed: `pip list | grep fastmcp`
3. Qdrant is running: `curl localhost:6333/health`

### pip broken in venv

**Error:** `bad interpreter: No such file or directory` when running pip

**Cause:** venv was moved or Python was upgraded.

**Fix:** Run `make setup` — it recreates the venv and reinstalls everything.

---

## Performance Tips

- **Narrow queries** work better than broad ones ("What does chapter 3 say about X?" vs "Tell me about X")
- **Filter by book** when you know which document contains the answer
- **Re-index after model changes** — wrong dimensions cause silent failures

---

## Useful Commands

```bash
# List indexed books with chunk counts
python src/ingest.py --list

# Check Qdrant health
curl localhost:6333/health

# Test API locally
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test query"}'

# Rebuild a specific book
python src/ingest.py --reindex investor-tom1

# Delete a book
python src/ingest.py --delete investor-tom1
```
