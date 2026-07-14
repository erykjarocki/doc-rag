# Testing

DOC-RAG uses a three-layer testing strategy to ensure correctness at every level — from pure utility functions to the full retrieval pipeline.

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

Full pipeline with real embeddings and in-memory Qdrant. Uses the [enterprise-rag-gold-standard](https://github.com/SantiagoCompany/enterprise-rag-gold-standard) benchmark corpus — 9 enterprise markdown documents with 8 ground-truth queries. Marked `@pytest.mark.eval`.

```bash
make test-eval
# or directly:
pytest tests/eval/ -v -m eval
# with reranking:
pytest tests/eval/ -v -m "eval or rerank"
```

**What's covered:**
- Real `multilingual-e5-small` embedding model
- Qdrant upsert and cosine similarity retrieval
- Retrieval quality across 8 labeled queries
- Citation formatting with Polish source labels
- **Quantified metrics:** Recall@2, Precision@2, MRR over labeled queries
- **Two-stage pipeline comparison:** Bi-encoder → Cross-encoder reranking

#### Benchmark corpus

The eval tests use a curated enterprise corpus (`tests/eval/benchmark_docs/`) from the enterprise-rag-gold-standard project:

| Document | Domain |
|----------|--------|
| `OP-204_Incident_Management_v4.md` | Operations |
| `BOM-FLT-H2NEX.md` | Supply Chain |
| `RGM-2026_Regional_Governance_Matrix.md` | Governance |
| `FIN-AUTH-101_Authorization_Framework.md` | Finance |
| `HR-POL-030_Remote_Work_US.md` | HR |
| `AVL-PUR-012_Supplier_Qualification.md` | Supply Chain |
| `SOP-COM-015_Regulatory_Breach_Report.md` | Compliance |
| `HR-POL-030-EMEA_Remote_Work.md` | HR |
| `LEG-GDPR-007_Supplier_Data_Processing.md` | Legal |

**Why this corpus?**
- Real enterprise documents with specific factual content (thresholds, policies, names)
- Small enough for fast CI (~40KB total, ~30 chunks)
- Verifiable ground truth from source_documents field
- Multi-document queries test cross-file retrieval
- No PDF extraction needed — pure markdown

#### Labels

`tests/eval/benchmark_labels.json` contains 8 queries mapped to source documents:

```json
{
  "query": "What is the financial threshold for Critical Exception?",
  "relevant_documents": ["OP-204_Incident_Management_v4.md"],
  "category": "direct_retrieval"
}
```

Relevance is **file-based**: a chunk is relevant if it comes from one of the `relevant_documents`.

#### Metrics and thresholds

| Metric | What it measures | Threshold | Why this threshold |
|--------|-----------------|-----------|-------------------|
| **Recall@2** | Did the relevant document appear in top-2? | >= 0.6 | With 9 documents, some queries may have overlapping content; 0.6 allows for imperfect retrieval |
| **Precision@2** | Are top-2 results from relevant documents? | >= 0.4 | At least 1 of top-2 should be from the right document |
| **MRR** | How high up is the first relevant result? | >= 0.5 | Average rank ~2 for first relevant result |

#### How scores are calculated

For a query with `relevant_documents = ["OP-204_Incident_Management_v4.md"]` and top-2 results from files `[OP-204_Incident_Management_v4.md, BOM-FLT-H2NEX.md]`:

```
Recall@2     = |relevant found in top-k| / |total relevant|  = 1/1 = 1.00
Precision@2  = |relevant found in top-k| / k                 = 1/2 = 0.50
MRR          = 1 / rank of first relevant result              = 1/1 = 1.00
```

Each query is evaluated independently, then metrics are averaged across all queries. The `pytest_sessionfinish` hook in `tests/eval/conftest.py` collects per-query results, computes averages, prints a terminal summary, and writes `tests/eval/eval-report.json`.

#### Two-stage pipeline comparison

The `@pytest.mark.rerank` tests run the full pipeline twice:

1. **Bi-encoder only** — baseline retrieval metrics
2. **Bi-encoder + cross-encoder** — reranked metrics

Both stages are compared in a single table: Before → After → Delta. The reranking detail shows per-query how items were reordered.

#### Configuration sweep

Compare retrieval quality across different configurations (chunk sizes, overlap, reranking):

```bash
make eval-sweep             # sequential (5 configs × ~90s each ≈ 7 min)
make eval-sweep-parallel    # parallel (all configs at once ≈ 2 min)
python tests/eval/run_sweep.py --configs chunk256 chunk512   # specific configs
python tests/eval/run_sweep.py --dry-run                     # preview without running
```

Default configurations:

| Config | Chunk Size | Overlap | Description |
|--------|-----------|---------|-------------|
| `chunk256` | 256 | 30 | Small chunks — faster, more precise |
| `chunk384` | 384 | 50 | Default — balanced |
| `chunk512` | 512 | 80 | Large chunks — more context per result |
| `overlap100` | 384 | 100 | High overlap — more context continuity |
| `no-rerank` | 384 | 50 | No cross-encoder reranking |

Results are saved to `tests/eval/eval-sweep/` with per-config reports and a comparison HTML at `sweep-comparison.html`. Add custom configs by editing the `DEFAULT_CONFIGS` dict in `tests/eval/run_sweep.py`.

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
    ├── conftest.py             # benchmark corpus fixtures, metric functions
    ├── benchmark_docs/         # enterprise-rag-gold-standard markdown docs
    ├── benchmark_labels.json   # 8 labeled queries (file-based relevance)
    ├── eval-baseline.json      # baseline metrics for regression detection
    ├── compare_to_baseline.py  # diffs current vs baseline scores
    ├── generate_report.py      # custom HTML report from eval-report.json
    ├── run_sweep.py            # multi-config sweep runner
    ├── sweep_report.py         # sweep comparison HTML generator
    └── test_e2e_pipeline.py    # E2E: retrieval + reranking (25 tests)
```

## How Mocking Works

The key challenge in testing a RAG system is avoiding expensive/slow operations (embedding model, vector DB). DOC-RAG handles this two ways:

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
pytest tests/eval/ -v -m "eval or rerank"
```

## Adding New Tests

1. **Pure function?** → Add to `tests/unit/test_<module>.py`, mark `@pytest.mark.unit`
2. **Needs Qdrant?** → Use the `qdrant_memory` fixture, mark `@pytest.mark.integration`
3. **Needs embeddings?** → Mock `get_model()` with a `MagicMock` that returns deterministic vectors
4. **Needs FastAPI?** → Use `TestClient(app)` from `fastapi.testclient`
5. **Full pipeline with real model?** → Add to `tests/eval/`, mark `@pytest.mark.eval`, use the `benchmark_indexed_qdrant` fixture from `tests/eval/conftest.py`

### Eval Test Design Philosophy

The eval tests follow a principle: **test the pipeline, not the model**. The embedding model is a black box — we don't test whether `multilingual-e5-small` produces optimal vectors. We test that:

1. The pipeline correctly wires chunking → embedding → storage → retrieval
2. The model's output is used correctly (vectors stored, queried, ranked)
3. Retrieval quality is *good enough* for the tool to be useful

This means:
- **Real enterprise documents** — not synthetic content. We control the ground truth via labeled queries.
- **Real model, not mocked** — mocked embeddings would test nothing. The model must actually distinguish incident management from valve specifications.
- **Quantified metrics** — not just "does it return something". Recall@k, Precision@k, MRR give a number you can track over time.
- **Strict thresholds** — on a small corpus, the model should be near-perfect. Loosening thresholds hides regressions.

### Adding Benchmark Queries

Add entries to `tests/eval/benchmark_labels.json`:

```json
{
  "query": "your question",
  "relevant_documents": ["filename.md"],
  "category": "direct_retrieval",
  "description": "why this document is relevant"
}
```

For multi-document queries:
```json
{
  "query": "question requiring two documents",
  "relevant_documents": ["doc1.md", "doc2.md"],
  "category": "multi_source",
  "description": "cross-document query"
}
```

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
