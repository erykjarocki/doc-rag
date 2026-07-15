import json
import os
import sys
from pathlib import Path

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import src.config as config
import src.ingest as ingest
import src.qdrant_store as qdrant_store

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.chunking import chunk_markdown
from src.embeddings import embed
from src.qdrant_store import ensure_collection

BENCHMARK_COLLECTION = "eval_benchmark"
BENCHMARK_DOCS_DIR = Path(__file__).parent / "benchmark_docs"

REPORT_PATH = Path(os.environ.get("EVAL_REPORT_PATH", Path(__file__).parent / "eval-report.json"))
BASELINE_PATH = Path(__file__).parent / "eval-baseline.json"


def pytest_sessionfinish(session, exitstatus):
    """Print terminal summary and write eval-report.json after all tests."""
    results = getattr(session, "eval_results", [])
    rerank_results = getattr(session, "rerank_results", [])
    if not results and not rerank_results:
        return

    def _compute_metrics(items):
        return {
            "recall_at_2": round(sum(i["recall_at_2"] for i in items) / len(items), 4),
            "precision_at_2": round(sum(i["precision_at_2"] for i in items) / len(items), 4),
            "mrr": round(sum(i["reciprocal_rank"] for i in items) / len(items), 4),
            "recall_at_4": round(sum(i["recall_at_4"] for i in items) / len(items), 4),
            "precision_at_4": round(sum(i["precision_at_4"] for i in items) / len(items), 4),
        }

    # Terminal summary
    terminal = session.config.get_terminal_writer()
    terminal.write("\n")
    terminal.write("=" * 70 + "\n")
    terminal.write("EVAL RESULTS\n")
    terminal.write("=" * 70 + "\n\n")

    for item in results:
        terminal.write(f'Query: "{item["query"]}"\n')
        for frag in item["retrieved_fragments"]:
            relevant = frag["is_relevant"]
            mark = "  \u2713 RELEVANT" if relevant else ""
            terminal.write(
                f"  [{frag['rank']}] score={frag['score']:.2f}  file={frag['source_file']}{mark}\n"
            )
            for line in frag["text"].split("\n"):
                terminal.write(f"      {line}\n")
            terminal.write("\n")
        terminal.write("\n")

    terminal.write("-" * 70 + "\n")

    if results:
        m = _compute_metrics(results)
        terminal.write(
            f"Recall@2: {m['recall_at_2']:.2f} | "
            f"Recall@4: {m['recall_at_4']:.2f} | "
            f"Precision@2: {m['precision_at_2']:.2f} | "
            f"Precision@4: {m['precision_at_4']:.2f} | "
            f"MRR: {m['mrr']:.2f}\n"
        )

    # Show baseline delta if available
    baseline = None
    if BASELINE_PATH.exists():
        try:
            raw = json.loads(BASELINE_PATH.read_text())
            if "metrics" in raw:
                baseline = raw["metrics"]
            elif "recall_at_2" in raw:
                baseline = raw
        except (json.JSONDecodeError, KeyError):
            pass

    if baseline and results:
        m = _compute_metrics(results)

        def _fmt_delta(cur, base):
            diff = cur - base
            if abs(diff) < 0.005:
                return "= 0.00"
            sign = "+" if diff > 0 else ""
            return f"{sign}{diff:.2f}"

        terminal.write(
            f"          "
            f"(base: {_fmt_delta(m['recall_at_2'], baseline.get('recall_at_2', 0))} | "
            f"{_fmt_delta(m['precision_at_2'], baseline.get('precision_at_2', 0))} | "
            f"{_fmt_delta(m['mrr'], baseline.get('mrr', 0))})\n"
        )

    # Two-stage comparison if both stages present
    if rerank_results:
        m_before = (
            _compute_metrics(results)
            if results
            else {"recall_at_2": 0, "precision_at_2": 0,
                  "mrr": 0, "recall_at_4": 0, "precision_at_4": 0}
        )
        m_after = _compute_metrics(rerank_results)
        terminal.write("\n")
        terminal.write("=" * 70 + "\n")
        terminal.write("PIPELINE COMPARISON: Bi-Encoder \u2192 Cross-Encoder Reranking\n")
        terminal.write("=" * 70 + "\n")
        terminal.write(f"  {'Metric':<15} {'Bi-Encoder':>12} {'+Rerank':>12} {'Delta':>10}\n")
        terminal.write(f"  {'-' * 49}\n")
        for key, label in [
            ("recall_at_2", "Recall@2"),
            ("recall_at_4", "Recall@4"),
            ("precision_at_2", "Precision@2"),
            ("precision_at_4", "Precision@4"),
            ("mrr", "MRR"),
        ]:
            b = m_before.get(key, 0)
            a = m_after.get(key, 0)
            d = a - b
            sign = "+" if d > 0 else ""
            terminal.write(f"  {label:<15} {b:>12.2f} {a:>12.2f} {sign}{d:>9.2f}\n")
        terminal.write("-" * 70 + "\n")

    terminal.write("-" * 70 + "\n\n")

    # Write JSON report
    report = {
        "queries": results,
        "metrics": _compute_metrics(results) if results else {},
    }

    if rerank_results:
        report["rerank_queries"] = rerank_results
        report["rerank_metrics"] = _compute_metrics(rerank_results)
        report["pipeline_comparison"] = {
            "before": _compute_metrics(results) if results else {},
            "after": _compute_metrics(rerank_results),
        }

    rerank_detail = getattr(session, "rerank_detail", [])
    if rerank_detail:
        report["rerank_detail"] = rerank_detail

    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # Generate detailed HTML report
    try:
        from tests.eval.generate_report import generate

        generate()
    except Exception:
        pass


def collect_eval_result(session, query, results, relevant_documents, k=2):
    """Run metrics on a query result and store on session for the summary hook.

    Relevance is file-based: a chunk is relevant if its source_file matches
    one of the relevant_documents listed in the label.
    """
    if not hasattr(session, "eval_results"):
        session.eval_results = []

    # Deduplicate: only store once per query
    if any(item["query"] == query for item in session.eval_results):
        return (
            _recall_at_k(results, relevant_documents, k),
            _precision_at_k(results, relevant_documents, k),
            _mrr(results, relevant_documents),
        )

    rr = _mrr(results, relevant_documents)

    top_k = results[:k]
    fragments = []
    for i, r in enumerate(top_k, 1):
        fragments.append(
            {
                "rank": i,
                "text": r["text"],
                "score": round(r["score"], 4),
                "source_file": r.get("source_file", ""),
                "is_relevant": r.get("source_file", "") in relevant_documents,
            }
        )

    session.eval_results.append(
        {
            "query": query,
            "relevant_documents": relevant_documents,
            "retrieved_fragments": fragments,
            "recall_at_k": _recall_at_k(results, relevant_documents, k),
            "precision_at_k": _precision_at_k(results, relevant_documents, k),
            "reciprocal_rank": rr,
            "recall_at_2": _recall_at_k(results, relevant_documents, 2),
            "precision_at_2": _precision_at_k(results, relevant_documents, 2),
            "recall_at_4": _recall_at_k(results, relevant_documents, 4),
            "precision_at_4": _precision_at_k(results, relevant_documents, 4),
        }
    )

    return (
        _recall_at_k(results, relevant_documents, k),
        _precision_at_k(results, relevant_documents, k),
        rr,
    )


def collect_rerank_result(session, query, results, relevant_documents, k=2):
    """Store stage-1 (bi-encoder) results for two-stage pipeline comparison."""
    if not hasattr(session, "rerank_results"):
        session.rerank_results = []

    if any(item["query"] == query for item in session.rerank_results):
        return (
            _recall_at_k(results, relevant_documents, k),
            _precision_at_k(results, relevant_documents, k),
            _mrr(results, relevant_documents),
        )

    rr = _mrr(results, relevant_documents)

    top_k = results[:k]
    fragments = []
    for i, r in enumerate(top_k, 1):
        fragments.append(
            {
                "rank": i,
                "text": r["text"],
                "score": round(r["score"], 4),
                "source_file": r.get("source_file", ""),
                "is_relevant": r.get("source_file", "") in relevant_documents,
            }
        )

    session.rerank_results.append(
        {
            "query": query,
            "relevant_documents": relevant_documents,
            "retrieved_fragments": fragments,
            "recall_at_k": _recall_at_k(results, relevant_documents, k),
            "precision_at_k": _precision_at_k(results, relevant_documents, k),
            "reciprocal_rank": rr,
            "recall_at_2": _recall_at_k(results, relevant_documents, 2),
            "precision_at_2": _precision_at_k(results, relevant_documents, 2),
            "recall_at_4": _recall_at_k(results, relevant_documents, 4),
            "precision_at_4": _precision_at_k(results, relevant_documents, 4),
        }
    )

    return (
        _recall_at_k(results, relevant_documents, k),
        _precision_at_k(results, relevant_documents, k),
        rr,
    )


def collect_rerank_detail(
    session, query, bi_results, reranked_results, rank_changes, relevant_documents
):
    """Store per-query before/after reranking detail for the HTML report."""
    if not hasattr(session, "rerank_detail"):
        session.rerank_detail = []

    if any(item["query"] == query for item in session.rerank_detail):
        return

    def _build_frag(r, rank, is_ce=False):
        source = r.get("source_file", "?")
        return {
            "rank": rank,
            "source_file": source,
            "bi_score": round(r["score"], 4),
            "ce_score": round(r.get("rerank_score", 0), 4) if is_ce else None,
            "text_preview": r["text"][:200] + ("\u2026" if len(r["text"]) > 200 else ""),
            "is_relevant": source in relevant_documents,
        }

    bi_top8 = [_build_frag(r, i + 1) for i, r in enumerate(bi_results[:8])]
    reranked_top8 = [_build_frag(r, i + 1, is_ce=True) for i, r in enumerate(reranked_results[:8])]

    bi_by_rank = {}
    for i, r in enumerate(bi_results):
        bi_by_rank[i + 1] = r.get("source_file", "?")

    clean_changes = []
    if rank_changes:
        for rc in rank_changes:
            before_rank = rc.get("before", 1)
            source = bi_by_rank.get(before_rank, rc.get("page", "?"))
            clean_changes.append(
                {
                    "source_file": source,
                    "before": before_rank,
                    "after": rc["after"],
                    "delta": rc["delta"],
                    "bi_score": round(rc["bi_score"], 4),
                    "ce_score": round(rc["ce_score"], 4),
                }
            )

    session.rerank_detail.append(
        {
            "query": query,
            "relevant_documents": relevant_documents,
            "bi_encoder_top8": bi_top8,
            "reranked_top8": reranked_top8,
            "rank_changes": clean_changes,
        }
    )


def _precision_at_k(results, relevant_documents, k):
    top_k = results[:k]
    relevant = sum(1 for r in top_k if r.get("source_file", "") in relevant_documents)
    return relevant / k


def _recall_at_k(results, relevant_documents, k):
    if not relevant_documents:
        return 1.0
    top_k = results[:k]
    found = set(
        r.get("source_file", "")
        for r in top_k
        if r.get("source_file", "") in relevant_documents
    )
    return len(found) / len(relevant_documents)


def _mrr(results, relevant_documents):
    for i, r in enumerate(results, 1):
        if r.get("source_file", "") in relevant_documents:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Benchmark corpus fixtures (enterprise-rag-gold-standard markdown docs)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_corpus():
    """Load markdown documents from benchmark_docs/.

    Returns list of (filename, text_content) tuples.
    """
    docs = []
    for md_file in sorted(BENCHMARK_DOCS_DIR.glob("*.md")):
        text = md_file.read_text()
        if text.strip():
            docs.append((md_file.name, text))
    assert len(docs) > 0, "No benchmark documents found"
    return docs


@pytest.fixture(scope="module")
def benchmark_indexed_qdrant(benchmark_corpus, tmp_path_factory):
    """Chunk, embed, and index benchmark docs into in-memory Qdrant.

    Each chunk stores source_file in payload for file-based relevance.
    Returns (qdrant_client, collection_name, chunks).
    """
    tmp_dir = tmp_path_factory.mktemp("benchmark")

    in_memory_client = QdrantClient(":memory:")
    original_client = qdrant_store._client

    qdrant_store._client = in_memory_client
    original_extracted = config.EXTRACTED_DIR
    config.EXTRACTED_DIR = str(tmp_dir / "extracted")
    ingest.EXTRACTED_DIR = config.EXTRACTED_DIR

    try:
        ensure_collection(BENCHMARK_COLLECTION, in_memory_client)

        all_chunks = []
        for filename, text in benchmark_corpus:
            chunks = chunk_markdown(text, source_file=filename)
            all_chunks.extend(chunks)

        assert len(all_chunks) > 0, "Expected at least one chunk from benchmark corpus"

        texts = [c["text"] for c in all_chunks]
        vectors = embed(texts)

        points = []
        for i, (chunk, vector) in enumerate(zip(all_chunks, vectors)):
            points.append(
                PointStruct(
                    id=i + 1,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "book": "eval_benchmark",
                        "source_file": chunk["source_file"],
                        "start_page": i + 1,
                        "end_page": i + 1,
                    },
                )
            )

        in_memory_client.upsert(collection_name=BENCHMARK_COLLECTION, points=points)
    except Exception:
        qdrant_store._client = original_client
        config.EXTRACTED_DIR = original_extracted
        ingest.EXTRACTED_DIR = original_extracted
        raise

    yield in_memory_client, BENCHMARK_COLLECTION, all_chunks

    qdrant_store._client = original_client
    config.EXTRACTED_DIR = original_extracted
    ingest.EXTRACTED_DIR = original_extracted
    try:
        in_memory_client.delete_collection(collection_name=BENCHMARK_COLLECTION)
    except Exception:
        pass
