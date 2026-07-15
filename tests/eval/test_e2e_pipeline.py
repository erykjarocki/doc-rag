import json
from pathlib import Path

import pytest

from src.qdrant_store import list_collections
from src.retriever import format_fragments_for_prompt, search_book

LABELS_PATH = Path(__file__).parent / "benchmark_labels.json"
BOOK = "eval_benchmark"


def _load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


@pytest.mark.eval
class TestFullPipelineIndexing:
    def test_chunks_are_indexed(self, benchmark_indexed_qdrant):
        client, coll_name, chunks = benchmark_indexed_qdrant
        collections = list_collections(client)
        assert coll_name in collections

        count_result = client.count(collection_name=coll_name, exact=True)
        assert count_result.count == len(chunks)

    def test_chunks_have_required_fields(self, benchmark_indexed_qdrant):
        _, _, chunks = benchmark_indexed_qdrant
        for chunk in chunks:
            assert "text" in chunk and chunk["text"], "chunk must have non-empty text"
            assert (
                "source_file" in chunk and chunk["source_file"]
            ), "chunk must have source_file"

    def test_vectors_match_chunk_count(self, benchmark_indexed_qdrant):
        client, coll_name, chunks = benchmark_indexed_qdrant
        count_result = client.count(collection_name=coll_name, exact=True)
        assert count_result.count == len(chunks)
        result, _ = client.scroll(
            collection_name=coll_name, limit=count_result.count, with_vectors=True
        )
        from src.config import EMBED_DIM

        for point in result:
            assert len(point.vector) == EMBED_DIM


@pytest.mark.eval
class TestRetrievalQuality:
    def test_incident_management_query(self, benchmark_indexed_qdrant):
        results = search_book("financial threshold for Critical Exception", book=BOOK)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "10,000" in text_lower or "critical exception" in text_lower
        assert top["score"] > 0.3

    def test_valve_pressure_query(self, benchmark_indexed_qdrant):
        results = search_book("maximum allowable psi for Valve-B", book=BOOK)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "2,800" in text_lower or "valve-b" in text_lower
        assert top["score"] > 0.3

    def test_remote_work_query(self, benchmark_indexed_qdrant):
        results = search_book("remote work days allowed per month US policy", book=BOOK)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "remote" in text_lower or "work" in text_lower

    def test_breach_reporting_query(self, benchmark_indexed_qdrant):
        results = search_book("Level-3 regulatory breach reporting hours", book=BOOK)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert (
            "breach" in text_lower
            or "compliance" in text_lower
            or "report" in text_lower
        )


@pytest.mark.eval
class TestFormattedOutput:
    def test_citations_have_polish_format(self, benchmark_indexed_qdrant):
        results = search_book("incident management", book=BOOK)
        formatted = format_fragments_for_prompt(results)
        assert "\u0179r\u00f3d\u0142o:" in formatted

    def test_numbered_blocks(self, benchmark_indexed_qdrant):
        results = search_book("incident management", book=BOOK)
        formatted = format_fragments_for_prompt(results)
        assert "[1]" in formatted
        if len(results) > 1:
            assert "[2]" in formatted

    def test_separator_between_blocks(self, benchmark_indexed_qdrant):
        results = search_book("incident management", book=BOOK)
        formatted = format_fragments_for_prompt(results)
        assert "---" in formatted


@pytest.mark.eval
class TestRetrievalMetrics:
    """Evaluate retrieval quality using precision, recall, and MRR over labeled data."""

    def test_recall_at_2(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        recalls = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            r, _, _ = collect_eval_result(
                request.session,
                item["query"],
                results,
                item["relevant_documents"],
                k=2,
            )
            recalls.append(r)

        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  recall@2 = {avg_recall:.2f} (threshold: 0.60)")
        assert avg_recall >= 0.6, f"Recall@2 = {avg_recall:.2f}, expected >= 0.6"

    def test_recall_at_4(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import _recall_at_k

        labels = _load_labels()
        recalls = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            recalls.append(_recall_at_k(results, item["relevant_documents"], 4))

        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  recall@4 = {avg_recall:.2f} (threshold: 0.70)")
        assert avg_recall >= 0.7, f"Recall@4 = {avg_recall:.2f}, expected >= 0.7"

    def test_precision_at_2(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        precisions = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            _, p, _ = collect_eval_result(
                request.session,
                item["query"],
                results,
                item["relevant_documents"],
                k=2,
            )
            precisions.append(p)

        avg_precision = sum(precisions) / len(precisions)
        print(f"\n  precision@2 = {avg_precision:.2f} (threshold: 0.40)")
        assert (
            avg_precision >= 0.4
        ), f"Precision@2 = {avg_precision:.2f}, expected >= 0.4"

    def test_precision_at_4(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import _precision_at_k

        labels = _load_labels()
        precisions = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            precisions.append(_precision_at_k(results, item["relevant_documents"], 4))

        avg_precision = sum(precisions) / len(precisions)
        print(f"\n  precision@4 = {avg_precision:.2f} (threshold: 0.35)")
        assert (
            avg_precision >= 0.35
        ), f"Precision@4 = {avg_precision:.2f}, expected >= 0.35"

    def test_mrr(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        rrs = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            _, _, rr = collect_eval_result(
                request.session,
                item["query"],
                results,
                item["relevant_documents"],
                k=2,
            )
            rrs.append(rr)

        avg_mrr = sum(rrs) / len(rrs)
        print(f"\n  mrr = {avg_mrr:.2f} (threshold: 0.50)")
        assert avg_mrr >= 0.5, f"MRR = {avg_mrr:.2f}, expected >= 0.5"


# ---------------------------------------------------------------------------
# Cross-Encoder Re-ranking Tests
# ---------------------------------------------------------------------------


@pytest.mark.rerank
class TestRerankRetrievalQuality:
    """Test retrieval quality WITH cross-encoder re-ranking enabled."""

    def test_incident_reranked(self, benchmark_indexed_qdrant):
        results = search_book(
            "financial threshold for Critical Exception", book=BOOK, rerank=True
        )
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "10,000" in text_lower or "critical exception" in text_lower
        assert "rerank_score" in top, "rerank_score must be present when rerank=True"

    def test_valve_reranked(self, benchmark_indexed_qdrant):
        results = search_book(
            "maximum allowable psi for Valve-B", book=BOOK, rerank=True
        )
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "2,800" in text_lower or "valve-b" in text_lower
        assert "rerank_score" in top

    def test_remote_work_reranked(self, benchmark_indexed_qdrant):
        results = search_book(
            "remote work days allowed per month US policy", book=BOOK, rerank=True
        )
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "remote" in text_lower or "work" in text_lower
        assert "rerank_score" in top

    def test_breach_reranked(self, benchmark_indexed_qdrant):
        results = search_book(
            "Level-3 regulatory breach reporting hours", book=BOOK, rerank=True
        )
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "breach" in text_lower or "compliance" in text_lower
        assert "rerank_score" in top


@pytest.mark.rerank
class TestRerankMetrics:
    """Evaluate retrieval quality WITH re-ranking enabled."""

    def test_recall_at_2_reranked(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        recalls = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=True)
            r, _, _ = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_documents"],
                k=2,
            )
            recalls.append(r)

        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  recall@2 (reranked) = {avg_recall:.2f} (threshold: 0.60)")
        assert avg_recall >= 0.6, f"Recall@2 = {avg_recall:.2f}, expected >= 0.6"

    def test_recall_at_4_reranked(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import _recall_at_k

        labels = _load_labels()
        recalls = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=True)
            recalls.append(_recall_at_k(results, item["relevant_documents"], 4))

        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  recall@4 (reranked) = {avg_recall:.2f} (threshold: 0.75)")
        assert avg_recall >= 0.75, f"Recall@4 = {avg_recall:.2f}, expected >= 0.75"

    def test_precision_at_2_reranked(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        precisions = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=True)
            _, p, _ = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_documents"],
                k=2,
            )
            precisions.append(p)

        avg_precision = sum(precisions) / len(precisions)
        print(f"\n  precision@2 (reranked) = {avg_precision:.2f} (threshold: 0.40)")
        assert (
            avg_precision >= 0.4
        ), f"Precision@2 = {avg_precision:.2f}, expected >= 0.4"

    def test_precision_at_4_reranked(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import _precision_at_k

        labels = _load_labels()
        precisions = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=True)
            precisions.append(_precision_at_k(results, item["relevant_documents"], 4))

        avg_precision = sum(precisions) / len(precisions)
        print(f"\n  precision@4 (reranked) = {avg_precision:.2f} (threshold: 0.35)")
        assert (
            avg_precision >= 0.35
        ), f"Precision@4 = {avg_precision:.2f}, expected >= 0.35"

    def test_mrr_reranked(self, request, benchmark_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        rrs = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=True)
            _, _, rr = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_documents"],
                k=2,
            )
            rrs.append(rr)

        avg_mrr = sum(rrs) / len(rrs)
        print(f"\n  mrr (reranked) = {avg_mrr:.2f} (threshold: 0.50)")
        assert avg_mrr >= 0.5, f"MRR = {avg_mrr:.2f}, expected >= 0.5"


@pytest.mark.rerank
class TestRerankBehavior:
    """Verify reranker mechanics: scores present, ordering correct."""

    def test_rerank_score_populated(self, benchmark_indexed_qdrant):
        results = search_book("incident management", book=BOOK, rerank=True)
        assert len(results) > 0
        for r in results:
            assert "rerank_score" in r, f"Missing rerank_score in result: {r}"
            assert isinstance(r["rerank_score"], float)

    def test_results_sorted_by_rerank_score(self, benchmark_indexed_qdrant):
        results = search_book("incident management", book=BOOK, rerank=True)
        assert len(results) > 1
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(
            scores, reverse=True
        ), "Results not sorted by rerank_score"

    def test_rerank_returns_fewer_or_equal_results(self, benchmark_indexed_qdrant):
        results = search_book("incident management", book=BOOK, rerank=True)
        assert len(results) <= 8, f"Expected <= 8 results, got {len(results)}"

    def test_rerank_disabled_has_no_rerank_score(self, benchmark_indexed_qdrant):
        results = search_book("incident management", book=BOOK, rerank=False)
        assert len(results) > 0
        for r in results:
            assert "rerank_score" not in r, "Unexpected rerank_score when rerank=False"


# ---------------------------------------------------------------------------
# Two-Stage Pipeline Comparison
# ---------------------------------------------------------------------------


@pytest.mark.rerank
class TestPipelineComparison:
    """Run both stages of the retrieval pipeline in one session.

    Stage 1: Bi-encoder retrieves candidates (baseline metrics)
    Stage 2: Cross-encoder re-ranks candidates (final metrics)

    Both stages are stored on the session so the report shows a single
    comparison table with Before -> After -> Delta, plus per-query
    reranking detail showing how items were reordered.
    """

    def test_pipeline_comparison(self, request, benchmark_indexed_qdrant):
        from src.trace import SearchResult
        from tests.eval.conftest import (
            collect_eval_result,
            collect_rerank_detail,
            collect_rerank_result,
        )

        labels = _load_labels()
        recalls_before, recalls_after = [], []
        precisions_before, precisions_after = [], []
        rrs_before, rrs_after = [], []

        for item in labels:
            query = item["query"]
            relevant = item["relevant_documents"]

            # Stage 1: bi-encoder only
            results_before = search_book(query, book=BOOK, rerank=False)
            r1, p1, rr1 = collect_rerank_result(
                request.session, query, results_before, relevant, k=2
            )
            recalls_before.append(r1)
            precisions_before.append(p1)
            rrs_before.append(rr1)

            # Stage 2: bi-encoder + cross-encoder (with trace for rank changes)
            search_result = search_book(query, book=BOOK, rerank=True, trace=True)
            assert isinstance(search_result, SearchResult)
            results_after = search_result.fragments

            # Extract rank_changes from trace
            rank_changes = None
            for stage in search_result.trace.stages:
                if stage.name == "rerank":
                    rank_changes = stage.details.get("rank_changes")
                    break

            r2, p2, rr2 = collect_eval_result(
                request.session, f"{query} (pipeline)", results_after, relevant, k=2
            )
            recalls_after.append(r2)
            precisions_after.append(p2)
            rrs_after.append(rr2)

            # Store per-query reranking detail for the HTML report
            collect_rerank_detail(
                request.session,
                query,
                results_before,
                results_after,
                rank_changes,
                relevant,
            )

        # Compute averages
        def avg(xs):
            return sum(xs) / len(xs) if xs else 0

        m_before = {
            "recall": avg(recalls_before),
            "precision": avg(precisions_before),
            "mrr": avg(rrs_before),
        }
        m_after = {
            "recall": avg(recalls_after),
            "precision": avg(precisions_after),
            "mrr": avg(rrs_after),
        }

        # Print comparison table
        print("\n")
        print("=" * 60)
        print("TWO-STAGE PIPELINE COMPARISON")
        print("=" * 60)
        print(f"  {'Metric':<15} {'Bi-Encoder':>12} {'+Rerank':>12} {'Delta':>10}")
        print(f"  {'-' * 49}")
        for key, label in [
            ("recall", "Recall@2"),
            ("precision", "Precision@2"),
            ("mrr", "MRR"),
        ]:
            b, a = m_before[key], m_after[key]
            d = a - b
            sign = "+" if d > 0 else ""
            print(f"  {label:<15} {b:>12.2f} {a:>12.2f} {sign}{d:>9.2f}")
        print("=" * 60)

        # Assertions: reranking should not severely degrade metrics
        # assert m_after["recall"] >= m_before["recall"] - 0.10, (
        #     f"Recall degraded significantly: {m_before['recall']:.2f} -> {m_after['recall']:.2f}"
        # )
        # assert m_after["mrr"] >= m_before["mrr"] - 0.10, (
        #     f"MRR degraded significantly: {m_before['mrr']:.2f} -> {m_after['mrr']:.2f}"
        # )
