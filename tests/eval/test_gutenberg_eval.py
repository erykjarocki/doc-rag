"""Gutenberg-based eval: retrieval quality with real content (The Prince, 26 chapters)."""

import json
from pathlib import Path

import pytest

from src.retriever import search_book

LABELS_PATH = Path(__file__).parent / "labels.json"
BOOK = "gutenberg_prince"


def _load_labels():
    with open(LABELS_PATH) as f:
        return json.load(f)


@pytest.mark.eval
class TestGutenbergRetrievalQuality:
    """Spot-check retrieval on specific chapters."""

    def test_mercenaries_query_finds_chapter_12(self, gutenberg_indexed_qdrant):
        results = search_book("dangers of mercenary soldiers", book=BOOK, rerank=False)
        assert len(results) > 0
        top = results[0]
        assert top["start_page"] == 12
        assert top["score"] > 0.3

    def test_fortune_query_finds_chapter_25(self, gutenberg_indexed_qdrant):
        results = search_book("what fortune can effect in human affairs", book=BOOK, rerank=False)
        assert len(results) > 0
        top = results[0]
        assert top["start_page"] == 25
        assert top["score"] > 0.3

    def test_flattery_query_finds_chapter_23(self, gutenberg_indexed_qdrant):
        results = search_book("how flatterers should be avoided", book=BOOK, rerank=False)
        assert len(results) > 0
        top = results[0]
        assert top["start_page"] == 23
        assert top["score"] > 0.3

    def test_war_query_finds_chapter_14(self, gutenberg_indexed_qdrant):
        results = search_book("war as the prince's only profession", book=BOOK, rerank=False)
        assert len(results) > 0
        top = results[0]
        assert top["start_page"] == 14
        assert top["score"] > 0.3

    def test_cross_topic_military_finds_multiple_chapters(
        self, gutenberg_indexed_qdrant
    ):
        results = search_book(
            "mercenary armies and military organization", book=BOOK, rerank=False
        )
        assert len(results) > 1
        pages = {r["start_page"] for r in results[:3]}
        assert len(pages) >= 2, f"Expected 2+ distinct chapters, got {pages}"


@pytest.mark.eval
class TestGutenbergRetrievalMetrics:
    """Evaluate retrieval quality using precision, recall, and MRR over labeled data."""

    def test_recall_at_2(self, request, gutenberg_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        recalls = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            r, _, _ = collect_eval_result(
                request.session,
                item["query"],
                results,
                item["relevant_pages"],
                k=2,
            )
            recalls.append(r)

        avg_recall = sum(recalls) / len(recalls)
        print(f"\n  recall@2 = {avg_recall:.2f} (threshold: 0.70)")
        assert avg_recall >= 0.7, f"Recall@2 = {avg_recall:.2f}, expected >= 0.7"

    def test_precision_at_2(self, request, gutenberg_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        precisions = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            _, p, _ = collect_eval_result(
                request.session,
                item["query"],
                results,
                item["relevant_pages"],
                k=2,
            )
            precisions.append(p)

        avg_precision = sum(precisions) / len(precisions)
        print(f"\n  precision@2 = {avg_precision:.2f} (threshold: 0.50)")
        assert avg_precision >= 0.5, (
            f"Precision@2 = {avg_precision:.2f}, expected >= 0.5"
        )

    def test_mrr(self, request, gutenberg_indexed_qdrant):
        from tests.eval.conftest import collect_eval_result

        labels = _load_labels()
        rrs = []

        for item in labels:
            results = search_book(item["query"], book=BOOK, rerank=False)
            _, _, rr = collect_eval_result(
                request.session,
                item["query"],
                results,
                item["relevant_pages"],
                k=2,
            )
            rrs.append(rr)

        avg_mrr = sum(rrs) / len(rrs)
        print(f"\n  mrr = {avg_mrr:.2f} (threshold: 0.60)")
        assert avg_mrr >= 0.6, f"MRR = {avg_mrr:.2f}, expected >= 0.6"
