import json
from pathlib import Path

import pytest

from src.qdrant_store import list_collections
from src.retriever import format_fragments_for_prompt, search_book

LABELS_PATH = Path(__file__).parent / "labels.json"
BOOK = "gutenberg_prince"


def _load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


@pytest.mark.eval
class TestFullPipelineIndexing:
    def test_chunks_are_indexed(self, gutenberg_indexed_qdrant):
        client, coll_name, chunks = gutenberg_indexed_qdrant
        collections = list_collections(client)
        assert coll_name in collections

        count_result = client.count(collection_name=coll_name, exact=True)
        assert count_result.count == len(chunks)

    def test_chunks_have_required_fields(self, gutenberg_indexed_qdrant):
        _, _, chunks = gutenberg_indexed_qdrant
        for chunk in chunks:
            assert "text" in chunk and chunk["text"], "chunk must have non-empty text"
            assert "book" in chunk and chunk["book"], "chunk must have non-empty book"
            assert "start_page" in chunk and chunk["start_page"] >= 1
            assert "end_page" in chunk and chunk["end_page"] >= chunk["start_page"]

    def test_vectors_match_chunk_count(self, gutenberg_indexed_qdrant):
        client, coll_name, chunks = gutenberg_indexed_qdrant
        result, _ = client.scroll(collection_name=coll_name, limit=200, with_vectors=True)
        assert len(result) == len(chunks)
        for point in result:
            assert len(point.vector) == 384


@pytest.mark.eval
class TestRetrievalQuality:
    def test_principalities_query_retrieves_chapter_1(self, gutenberg_indexed_qdrant):
        results = search_book("different kinds of principalities", book=BOOK)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "principalit" in text_lower
        assert top["score"] > 0.3

    def test_darius_query_retrieves_chapter_4(self, gutenberg_indexed_qdrant):
        results = search_book("Darius and Alexander conquest", book=BOOK)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "darius" in text_lower or "alexander" in text_lower
        assert top["score"] > 0.3

    def test_feared_vs_loved_query(self, gutenberg_indexed_qdrant):
        results = search_book("being feared versus being loved", book=BOOK)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "fear" in text_lower or "love" in text_lower or "hate" in text_lower

    def test_mercenaries_query(self, gutenberg_indexed_qdrant):
        results = search_book("dangers of mercenary soldiers", book=BOOK)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "mercenary" in text_lower or "soldier" in text_lower or "army" in text_lower

    def test_fortune_query(self, gutenberg_indexed_qdrant):
        results = search_book("fortune and luck in ruling", book=BOOK)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "fortune" in text_lower or "luck" in text_lower


@pytest.mark.eval
class TestFormattedOutput:
    def test_citations_have_polish_format(self, gutenberg_indexed_qdrant):
        results = search_book("principalities", book=BOOK)
        formatted = format_fragments_for_prompt(results)
        assert "\u0179r\u00f3d\u0142o:" in formatted
        assert "str." in formatted

    def test_numbered_blocks(self, gutenberg_indexed_qdrant):
        results = search_book("principalities", book=BOOK)
        formatted = format_fragments_for_prompt(results)
        assert "[1]" in formatted
        if len(results) > 1:
            assert "[2]" in formatted

    def test_separator_between_blocks(self, gutenberg_indexed_qdrant):
        results = search_book("principalities", book=BOOK)
        formatted = format_fragments_for_prompt(results)
        assert "---" in formatted


@pytest.mark.eval
class TestRetrievalMetrics:
    """Evaluate retrieval quality using precision, recall, and MRR over labeled data."""

    def test_recall_at_2(self, request, gutenberg_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        recalls = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            r, _, _ = collect_eval_result(
                request.session, item["query"], results, item["relevant_pages"], k=2
            )
            recalls.append(r)

        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  recall@2 = {avg_recall:.2f} (threshold: 0.60)")
        assert avg_recall >= 0.6, f"Recall@2 = {avg_recall:.2f}, expected >= 0.6"

    def test_precision_at_2(self, request, gutenberg_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        precisions = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            _, p, _ = collect_eval_result(
                request.session, item["query"], results, item["relevant_pages"], k=2
            )
            precisions.append(p)

        avg_precision = sum(precisions) / len(precisions)
        print(f"\n  precision@2 = {avg_precision:.2f} (threshold: 0.40)")
        assert avg_precision >= 0.4, f"Precision@2 = {avg_precision:.2f}, expected >= 0.4"

    def test_mrr(self, request, gutenberg_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        rrs = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            _, _, rr = collect_eval_result(
                request.session, item["query"], results, item["relevant_pages"], k=2
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

    def test_principalities_reranked(self, gutenberg_indexed_qdrant):
        results = search_book("different kinds of principalities", book=BOOK, rerank=True)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "principalit" in text_lower
        assert "rerank_score" in top, "rerank_score must be present when rerank=True"

    def test_darius_reranked(self, gutenberg_indexed_qdrant):
        results = search_book("Darius and Alexander conquest", book=BOOK, rerank=True)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "darius" in text_lower or "alexander" in text_lower
        assert "rerank_score" in top

    def test_feared_vs_loved_reranked(self, gutenberg_indexed_qdrant):
        results = search_book("being feared versus being loved", book=BOOK, rerank=True)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "fear" in text_lower or "love" in text_lower or "hate" in text_lower
        assert "rerank_score" in top

    def test_mercenaries_reranked(self, gutenberg_indexed_qdrant):
        results = search_book("dangers of mercenary soldiers", book=BOOK, rerank=True)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "mercenary" in text_lower or "soldier" in text_lower or "army" in text_lower
        assert "rerank_score" in top

    def test_fortune_reranked(self, gutenberg_indexed_qdrant):
        results = search_book("fortune and luck in ruling", book=BOOK, rerank=True)
        assert len(results) > 0
        top = results[0]
        text_lower = top["text"].lower()
        assert "fortune" in text_lower or "luck" in text_lower
        assert "rerank_score" in top


@pytest.mark.rerank
class TestRerankMetrics:
    """Evaluate retrieval quality WITH re-ranking enabled."""

    def test_recall_at_2_reranked(self, request, gutenberg_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        recalls = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=True)
            r, _, _ = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_pages"],
                k=2,
            )
            recalls.append(r)

        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  recall@2 (reranked) = {avg_recall:.2f} (threshold: 0.60)")
        assert avg_recall >= 0.6, f"Recall@2 = {avg_recall:.2f}, expected >= 0.6"

    def test_precision_at_2_reranked(self, request, gutenberg_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        precisions = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=True)
            _, p, _ = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_pages"],
                k=2,
            )
            precisions.append(p)

        avg_precision = sum(precisions) / len(precisions)
        print(f"\n  precision@2 (reranked) = {avg_precision:.2f} (threshold: 0.40)")
        assert avg_precision >= 0.4, f"Precision@2 = {avg_precision:.2f}, expected >= 0.4"

    def test_mrr_reranked(self, request, gutenberg_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        rrs = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=True)
            _, _, rr = collect_eval_result(
                request.session,
                f"{item['query']}_reranked",
                results,
                item["relevant_pages"],
                k=2,
            )
            rrs.append(rr)

        avg_mrr = sum(rrs) / len(rrs)
        print(f"\n  mrr (reranked) = {avg_mrr:.2f} (threshold: 0.50)")
        assert avg_mrr >= 0.5, f"MRR = {avg_mrr:.2f}, expected >= 0.5"


@pytest.mark.rerank
class TestRerankBehavior:
    """Verify reranker mechanics: scores present, ordering correct."""

    def test_rerank_score_populated(self, gutenberg_indexed_qdrant):
        results = search_book("principalities", book=BOOK, rerank=True)
        assert len(results) > 0
        for r in results:
            assert "rerank_score" in r, f"Missing rerank_score in result: {r}"
            assert isinstance(r["rerank_score"], float)

    def test_results_sorted_by_rerank_score(self, gutenberg_indexed_qdrant):
        results = search_book("principalities", book=BOOK, rerank=True)
        assert len(results) > 1
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not sorted by rerank_score"

    def test_rerank_returns_fewer_or_equal_results(self, gutenberg_indexed_qdrant):
        results = search_book("principalities", book=BOOK, rerank=True)
        assert len(results) <= 8, f"Expected <= 8 results, got {len(results)}"

    def test_rerank_disabled_has_no_rerank_score(self, gutenberg_indexed_qdrant):
        results = search_book("principalities", book=BOOK, rerank=False)
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
    comparison table with Before → After → Delta, plus per-query
    reranking detail showing how items were reordered.
    """

    def test_pipeline_comparison(self, request, gutenberg_indexed_qdrant):
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
            relevant = item["relevant_pages"]

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
        for key, label in [("recall", "Recall@2"), ("precision", "Precision@2"), ("mrr", "MRR")]:
            b, a = m_before[key], m_after[key]
            d = a - b
            sign = "+" if d > 0 else ""
            print(f"  {label:<15} {b:>12.2f} {a:>12.2f} {sign}{d:>9.2f}")
        print("=" * 60)

        # Assertions: reranking should not severely degrade metrics
        assert m_after["recall"] >= m_before["recall"] - 0.10, (
            f"Recall degraded significantly: {m_before['recall']:.2f} → {m_after['recall']:.2f}"
        )
        assert m_after["mrr"] >= m_before["mrr"] - 0.10, (
            f"MRR degraded significantly: {m_before['mrr']:.2f} → {m_after['mrr']:.2f}"
        )
