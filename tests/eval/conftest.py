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

from src.chunking import chunk_text
from src.embeddings import embed
from src.qdrant_store import ensure_collection

GUTENBERG_COLLECTION = "gutenberg_prince"

REPORT_PATH = Path(__file__).parent / "eval-report.json"
BASELINE_PATH = Path(__file__).parent / "eval-baseline.json"


def pytest_sessionfinish(session, exitstatus):
    """Print terminal summary and write eval-report.json after all tests."""
    results = getattr(session, "eval_results", [])
    rerank_results = getattr(session, "rerank_results", [])
    if not results and not rerank_results:
        return

    def _compute_metrics(items):
        recalls = [item["recall_at_k"] for item in items]
        precisions = [item["precision_at_k"] for item in items]
        rrs = [item["reciprocal_rank"] for item in items]
        return {
            "recall_at_2": round(sum(recalls) / len(recalls), 4),
            "precision_at_2": round(sum(precisions) / len(precisions), 4),
            "mrr": round(sum(rrs) / len(rrs), 4),
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
                f"  [{frag['rank']}] score={frag['score']:.2f}  page={frag['start_page']}{mark}\n"
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
            f"Precision@2: {m['precision_at_2']:.2f} | "
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
            else {"recall_at_2": 0, "precision_at_2": 0, "mrr": 0}
        )
        m_after = _compute_metrics(rerank_results)
        terminal.write("\n")
        terminal.write("=" * 70 + "\n")
        terminal.write("PIPELINE COMPARISON: Bi-Encoder → Cross-Encoder Reranking\n")
        terminal.write("=" * 70 + "\n")
        terminal.write(f"  {'Metric':<15} {'Before':>10} {'After':>10} {'Delta':>10}\n")
        terminal.write(f"  {'-' * 45}\n")
        for key, label in [
            ("recall_at_2", "Recall@2"),
            ("precision_at_2", "Precision@2"),
            ("mrr", "MRR"),
        ]:
            b = m_before[key]
            a = m_after[key]
            d = a - b
            sign = "+" if d > 0 else ""
            terminal.write(f"  {label:<15} {b:>10.2f} {a:>10.2f} {sign}{d:>9.2f}\n")
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


def collect_eval_result(session, query, results, relevant_pages, k=2):
    """Run metrics on a query result and store on session for the summary hook."""
    if not hasattr(session, "eval_results"):
        session.eval_results = []

    # Deduplicate: only store once per query
    if any(item["query"] == query for item in session.eval_results):
        top_k = results[:k]
        return (
            _recall_at_k(results, relevant_pages, k),
            _precision_at_k(results, relevant_pages, k),
            _mrr(results, relevant_pages),
        )

    top_k = results[:k]
    rr = _mrr(results, relevant_pages)
    recall = _recall_at_k(results, relevant_pages, k)
    precision = _precision_at_k(results, relevant_pages, k)

    fragments = []
    for i, r in enumerate(top_k, 1):
        fragments.append(
            {
                "rank": i,
                "text": r["text"],
                "score": round(r["score"], 4),
                "start_page": r["start_page"],
                "end_page": r["end_page"],
                "chapter": r.get("chapter", ""),
                "is_relevant": r["start_page"] in relevant_pages,
            }
        )

    session.eval_results.append(
        {
            "query": query,
            "relevant_pages": relevant_pages,
            "retrieved_fragments": fragments,
            "recall_at_k": recall,
            "precision_at_k": precision,
            "reciprocal_rank": rr,
        }
    )

    return recall, precision, rr


def collect_rerank_result(session, query, results, relevant_pages, k=2):
    """Store stage-1 (bi-encoder) results for two-stage pipeline comparison.

    Uses a separate session attribute (rerank_results) so both stages
    appear in the same report.
    """
    if not hasattr(session, "rerank_results"):
        session.rerank_results = []

    if any(item["query"] == query for item in session.rerank_results):
        return (
            _recall_at_k(results, relevant_pages, k),
            _precision_at_k(results, relevant_pages, k),
            _mrr(results, relevant_pages),
        )

    top_k = results[:k]
    rr = _mrr(results, relevant_pages)
    recall = _recall_at_k(results, relevant_pages, k)
    precision = _precision_at_k(results, relevant_pages, k)

    fragments = []
    for i, r in enumerate(top_k, 1):
        fragments.append(
            {
                "rank": i,
                "text": r["text"],
                "score": round(r["score"], 4),
                "start_page": r["start_page"],
                "end_page": r["end_page"],
                "chapter": r.get("chapter", ""),
                "is_relevant": r["start_page"] in relevant_pages,
            }
        )

    session.rerank_results.append(
        {
            "query": query,
            "relevant_pages": relevant_pages,
            "retrieved_fragments": fragments,
            "recall_at_k": recall,
            "precision_at_k": precision,
            "reciprocal_rank": rr,
        }
    )

    return recall, precision, rr


def collect_rerank_detail(
    session, query, bi_results, reranked_results, rank_changes, relevant_pages
):
    """Store per-query before/after reranking detail for the HTML report.

    Args:
        session: Pytest session object.
        query: The query string.
        bi_results: Bi-encoder results (top 8, from rerank=False).
        reranked_results: Cross-encoder results (top 8, from rerank=True).
        rank_changes: List of rank change dicts from trace, or None.
        relevant_pages: List of relevant page numbers.
    """
    if not hasattr(session, "rerank_detail"):
        session.rerank_detail = []

    if any(item["query"] == query for item in session.rerank_detail):
        return

    def _build_frag(r, rank, is_ce=False):
        return {
            "rank": rank,
            "page": r.get("start_page", "?"),
            "chapter": r.get("chapter", ""),
            "bi_score": round(r["score"], 4),
            "ce_score": round(r.get("rerank_score", 0), 4) if is_ce else None,
            "text_preview": r["text"][:200] + ("…" if len(r["text"]) > 200 else ""),
            "is_relevant": r.get("start_page", "?") in relevant_pages,
        }

    bi_top8 = [_build_frag(r, i + 1) for i, r in enumerate(bi_results[:8])]
    reranked_top8 = [_build_frag(r, i + 1, is_ce=True) for i, r in enumerate(reranked_results[:8])]

    # Build a lookup of bi_score by page for rank_changes augmentation
    bi_by_page = {}
    for r in bi_results:
        p = r.get("start_page", "?")
        if p not in bi_by_page:
            bi_by_page[p] = round(r["score"], 4)

    clean_changes = []
    if rank_changes:
        for rc in rank_changes:
            clean_changes.append(
                {
                    "page": rc["page"],
                    "chapter": rc.get("chapter", ""),
                    "before": rc["before"],
                    "after": rc["after"],
                    "delta": rc["delta"],
                    "bi_score": round(rc["bi_score"], 4),
                    "ce_score": round(rc["ce_score"], 4),
                }
            )

    session.rerank_detail.append(
        {
            "query": query,
            "relevant_pages": relevant_pages,
            "bi_encoder_top8": bi_top8,
            "reranked_top8": reranked_top8,
            "rank_changes": clean_changes,
        }
    )


def _precision_at_k(results, relevant_pages, k):
    top_k = results[:k]
    relevant = sum(1 for r in top_k if r["start_page"] in relevant_pages)
    return relevant / k


def _recall_at_k(results, relevant_pages, k):
    if not relevant_pages:
        return 1.0
    top_k = results[:k]
    found_pages = set(r["start_page"] for r in top_k if r["start_page"] in relevant_pages)
    return len(found_pages) / len(relevant_pages)


def _mrr(results, relevant_pages):
    for i, r in enumerate(results, 1):
        if r["start_page"] in relevant_pages:
            return 1.0 / i
    return 0.0


# ---------------------------------------------------------------------------
# Gutenberg corpus fixtures (real content, no PDF)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gutenberg_corpus():
    """Fetch The Prince from Gutenberg, split into chapters.

    Returns (text, page_boundaries, page_nums) — 26 chapters.
    Cached via lru_cache so multiple fixtures share one fetch.
    """
    from tests.eval.gutenberg_corpus import fetch_and_split

    return fetch_and_split()


@pytest.fixture(scope="module")
def gutenberg_indexed_qdrant(gutenberg_corpus, tmp_path_factory):
    """Chunk, embed, and index the Gutenberg corpus into in-memory Qdrant.

    Uses chunk_text() directly — no PDF parsing, no extraction.
    Returns (qdrant_client, collection_name, chunks).
    """
    text, page_boundaries, page_nums = gutenberg_corpus
    tmp_dir = tmp_path_factory.mktemp("gutenberg")

    in_memory_client = QdrantClient(":memory:")
    original_client = qdrant_store._client

    qdrant_store._client = in_memory_client
    original_extracted = config.EXTRACTED_DIR
    config.EXTRACTED_DIR = str(tmp_dir / "extracted")
    ingest.EXTRACTED_DIR = config.EXTRACTED_DIR

    try:
        ensure_collection(GUTENBERG_COLLECTION, in_memory_client)

        chunks = chunk_text(text, page_boundaries, page_nums)
        assert len(chunks) > 0, "Expected at least one chunk from Gutenberg corpus"

        texts = [c["text"] for c in chunks]
        vectors = embed(texts)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=i + 1,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "book": "gutenberg_prince",
                        "chapter": f"Chapter {chunk['start_page']}",
                        "start_page": chunk["start_page"],
                        "end_page": chunk["end_page"],
                    },
                )
            )

        in_memory_client.upsert(collection_name=GUTENBERG_COLLECTION, points=points)
    except Exception:
        qdrant_store._client = original_client
        config.EXTRACTED_DIR = original_extracted
        ingest.EXTRACTED_DIR = original_extracted
        raise

    yield in_memory_client, GUTENBERG_COLLECTION, chunks

    qdrant_store._client = original_client
    config.EXTRACTED_DIR = original_extracted
    ingest.EXTRACTED_DIR = original_extracted
    try:
        in_memory_client.delete_collection(collection_name=GUTENBERG_COLLECTION)
    except Exception:
        pass
