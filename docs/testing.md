# Testing

PDF-RAG uses a three-layer testing strategy to ensure correctness at every level — from pure utility functions to the full retrieval pipeline.

## Test Layers

### Unit Tests

Fast, no I/O. Test pure logic only — chunking, chapter detection, formatting, config.

```bash
make test-unit
# or directly:
pytest tests/unit/ -v -m unit
```

**What's covered:**
- `collection_name()` sanitization (special chars, casing, empty input)
- `detect_chapter()` regex for Polish and English headings
- `get_page_boundaries()` and `_page_at_position()` page tracking
- `format_fragments_for_prompt()` citation formatting

### Integration Tests

Use in-memory Qdrant (`QdrantClient(":memory:")`) and mocked embeddings. No Docker, no network.

```bash
make test-integration
# or directly:
pytest tests/integration/ -v -m integration
```

**What's covered:**
- Qdrant collection CRUD (create, delete, list)
- Vector upsert and cosine similarity query
- Score ranking (exact match ranks higher than partial)
- FastAPI endpoint contract (`/health`, `/query`, `/books`, `/delete`)

### Eval Tests

Full pipeline with real embeddings and in-memory Qdrant. Generates a tiny test PDF, runs extraction, chunking, chapter detection, embedding, and retrieval — no mocking. Marked `@pytest.mark.eval`.

```bash
make test-eval
# or directly:
pytest tests/eval/ -v -m eval
# with HTML report:
pytest tests/eval/ -v -m eval --html=eval-report.html --self-contained-html
```

**What's covered:**
- Full `process_book()` pipeline (extract → chunk → chapter detection)
- Real `multilingual-e5-small` embedding model
- Qdrant upsert and cosine similarity retrieval
- Retrieval quality: queries retrieve semantically correct chunks
- Citation formatting with Polish source labels
- **Quantified metrics:** Recall@2, Precision@2, MRR over labeled queries

**What you see in the terminal:**
```
tests/eval/test_e2e_pipeline.py::TestRetrievalMetrics::test_recall_at_2
  recall@2 = 1.00 (threshold: 0.80)
PASSED
...
======================================================================
EVAL RESULTS
======================================================================

Query: "What is the capital of France?"
  [1] score=0.85  page=1  ✓ RELEVANT
      Chapter 1: France
      Paris is the capital and most populous city of France...
  [2] score=0.83  page=1  ✓ RELEVANT
      ...Notre-Dame de Paris is a medieval Catholic cathedral...

Query: "Tell me about Berlin"
  [1] score=0.81  page=2  ✓ RELEVANT
      ...The Berlin Wall divided the city from August 13, 1961...
  [2] score=0.80  page=1
      ...Chapter 2: Germany...

----------------------------------------------------------------------
Recall@2: 1.00 | Precision@2: 0.83 | MRR: 1.00
----------------------------------------------------------------------
```

After the run, `tests/eval/eval-report.json` contains the full structured results with per-query fragment text, scores, and relevance flags.

#### Why a 3-topic synthetic PDF?

The test PDF is intentionally minimal — three pages, three topics (France/Germany/Japan), ~4600 characters total. This is deliberate:

- **Deterministic chunking.** A ~4600-char document produces ~7 chunks (with 384-token chunk size). This is enough to test retrieval discrimination without being overwhelming.
- **Fast feedback.** The model loads once (~1s), extraction is instant, embedding is a single batch. Total: ~8s. A real book would take minutes.
- **Focused assertions.** We're testing the *pipeline wiring*, not the model's knowledge. If "Paris", "Berlin", and "Tokyo" are in separate chunks and the model can't distinguish them, something is broken in the pipeline — not the model.
- **Easy to debug.** When a metric fails, you know exactly which chunk should have matched. No need to inspect 50 pages of output.
- **3 topics, not 2.** Two topics can be distinguished by simple keyword matching. Three topics require the model to actually understand semantic similarity — a better test of the embedding model.

This is standard practice for E2E pipeline tests. Real-world PDFs belong in manual evaluation, not automated CI.

#### Why not test with a real book?

Testing with a 300-page book would:
- Take 2-5 minutes per CI run (embedding model load + extraction + chunking)
- Make failures ambiguous (is the model wrong, or the chunking?)
- Require checking in a large PDF to git (or downloading it in CI)
- Create flaky tests (floating-point differences across platforms)

The tiny PDF catches the same bugs: broken extraction, wrong chunk boundaries, missing vectors, disconnected retrieval.

#### Metrics and thresholds

The test uses `tests/eval/labels.json` — 9 labeled queries (3 per topic) with expected relevant page numbers. Three metrics are computed:

| Metric | What it measures | Threshold | Why this threshold |
|--------|-----------------|-----------|-------------------|
| **Recall@2** | Did the relevant chunk appear in top-2? | >= 0.8 | With 7 chunks and 1 relevant per query, recall=1.0 means all relevant pages are found. 0.8 allows one query to miss. |
| **Precision@2** | Are top-2 results mostly relevant? | >= 0.5 | With 7 chunks, precision@2 can range from 0 (both irrelevant) to 1.0 (both relevant). 0.5 means at least half of top-2 are relevant. |
| **MRR** | How high up is the first relevant result? | >= 0.7 | MRR=1.0 means relevant result is always rank-1. 0.7 means average rank ~1.4, which is acceptable for a 7-chunk corpus. |

**Why these specific thresholds?** They're calibrated for a 7-chunk corpus where topics should be well-separated. If the model can't distinguish Paris from Berlin from Tokyo in a tiny PDF, it won't work on real books. The thresholds are intentionally strict — this is a sanity check, not a lenient pass.

**How to extend:** Add queries to `tests/eval/labels.json`. Each entry needs:
```json
{
  "query": "your question",
  "relevant_pages": [1],
  "description": "why this page is relevant"
}
```

As the corpus grows, tighten thresholds. With 20+ chunks, expect Precision@2 to drop — consider raising k or adding a Recall@5 metric.

#### How scores are calculated

For a query with `relevant_pages = [1]` and top-2 results with pages `[1, 2]`:

```
Recall@2     = |relevant found in top-k| / |total relevant|  = 1/1 = 1.00
Precision@2  = |relevant found in top-k| / k                 = 1/2 = 0.50
MRR          = 1 / rank of first relevant result              = 1/1 = 1.00
```

For a query where the relevant chunk is at rank-2: `MRR = 1/2 = 0.50`.

For a query with `relevant_pages = [3]` where both top-2 results are from page 3:

```
Recall@2     = 1/1 = 1.00
Precision@2  = 2/2 = 1.00
MRR          = 1/1 = 1.00
```

Each query is evaluated independently, then metrics are averaged across all queries. The `pytest_sessionfinish` hook in `tests/eval/conftest.py` collects per-query results, computes averages, prints a terminal summary, and writes `tests/eval/eval-report.json`.

The `collect_eval_result()` function in `tests/eval/conftest.py` computes per-query metrics and stores per-fragment details (chunk text, cosine score, rank, relevance flag) that feed into both the terminal output and JSON report. This gives full visibility into what the model retrieved for each query.

## Running All Tests

```bash
make test             # all layers
make test-unit        # unit only
make test-integration # integration only
make test-eval        # E2E pipeline (real model, ~30s)
```

## Test Structure

```
tests/
├── conftest.py                # shared fixtures (in-memory Qdrant, sample data)
├── unit/
│   ├── test_utils.py           # collection_name() sanitization
│   ├── test_chapter_detection.py  # ChapterDetector 3-layer strategy
│   ├── test_ingest.py          # page boundaries, extraction utils
│   └── test_retriever.py       # formatting, result parsing
├── integration/
│   ├── test_retrieval.py       # Qdrant round-trip (in-memory)
│   └── test_api.py             # FastAPI TestClient endpoint tests
└── eval/
    ├── conftest.py             # tiny PDF generator, full-pipeline fixture
    └── test_e2e_pipeline.py    # E2E: extract → embed → store → query → format
```

## How Mocking Works

The key challenge in testing a RAG system is avoiding expensive/slow operations (embedding model, vector DB). PDF-RAG handles this two ways:

1. **In-memory Qdrant** — `QdrantClient(":memory:")` runs the full vector DB in-process. No Docker needed. Tests that need Qdrant get a fresh client via the `qdrant_memory` fixture.

2. **Mocked embeddings** — `@patch("src.embeddings.get_model")` replaces the SentenceTransformer model with a `MagicMock`. This lets us test chunking and formatting logic without loading the 100MB+ model.

**Important:** Always patch where the function is *used*, not where it's defined. For example, `ingest.py` imports `get_model` from `src.embeddings`, so patch `"src.ingest.get_model"` — not `"src.embeddings.get_model"`.

## CI Integration

GitHub Actions runs a 3-stage pipeline on every PR (see `.github/workflows/ci.yml`):

- **Triggers:** Push/PR to `main` when `src/`, `tests/`, or `pyproject.toml` change
- **Python:** 3.12 (single version — project is a local tool, not a library)
- **Stage 1:** Install dependencies (warms pip cache)
- **Stage 2 (parallel):** Lint (ruff), Unit tests, Integration tests, Eval tests
- **Optimization:** CPU-only PyTorch installed first (~150MB vs ~3GB with CUDA)
- **No Docker needed** — all tests use in-memory Qdrant

### Testing CI Locally

```bash
# Run the same commands CI executes:
ruff check src/ tests/
pytest tests/unit/ -v -m unit
pytest tests/integration/ -v -m integration
pytest tests/eval/ -v -m eval
```

## Adding New Tests

1. **Pure function?** → Add to `tests/unit/test_<module>.py`, mark `@pytest.mark.unit`
2. **Needs Qdrant?** → Use the `qdrant_memory` fixture, mark `@pytest.mark.integration`
3. **Needs embeddings?** → Mock `get_model()` with a `MagicMock` that returns deterministic vectors
4. **Needs FastAPI?** → Use `TestClient(app)` from `fastapi.testclient`
5. **Full pipeline with real model?** → Add to `tests/eval/`, mark `@pytest.mark.eval`, use the `indexed_qdrant` fixture from `tests/eval/conftest.py`

### Eval Test Design Philosophy

The eval tests follow a principle: **test the pipeline, not the model**. The embedding model is a black box — we don't test whether `multilingual-e5-small` produces optimal vectors. We test that:

1. The pipeline correctly wires extraction → chunking → embedding → storage → retrieval
2. The model's output is used correctly (vectors stored, queried, ranked)
3. Retrieval quality is *good enough* for the tool to be useful

This means:
- **Small, predictable PDFs** — not real books. We control the ground truth.
- **Real model, not mocked** — mocked embeddings would test nothing. The model must actually separate Paris from Berlin.
- **Quantified metrics** — not just "does it return something". Recall@k, Precision@k, MRR give a number you can track over time.
- **Strict thresholds** — on a 2-chunk corpus, the model should be near-perfect. Loosening thresholds hides regressions.

### Example: Testing a New Function

```python
# tests/unit/test_example.py
import pytest

@pytest.mark.unit
def test_my_new_function():
    from src.my_module import my_new_function
    result = my_new_function("input")
    assert result == "expected"
```

```python
# tests/integration/test_example.py
import pytest
from unittest.mock import patch

@pytest.mark.integration
def test_with_qdrant(qdrant_memory):
    from src.qdrant_store import ensure_collection
    ensure_collection("test", qdrant_memory)
    assert "test" in qdrant_memory.get_collections().collections
```
