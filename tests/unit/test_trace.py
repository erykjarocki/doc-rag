import pytest

from src.trace import SearchResult, StageTrace, TraceLog, format_trace


@pytest.mark.unit
class TestStageTrace:
    def test_creation(self):
        stage = StageTrace(
            name="embed",
            input_summary="query text",
            output_summary="384d vector",
            duration_ms=2.5,
            details={"model": "test-model"},
        )
        assert stage.name == "embed"
        assert stage.duration_ms == 2.5
        assert stage.details["model"] == "test-model"

    def test_default_details(self):
        stage = StageTrace(
            name="retrieve",
            input_summary="in",
            output_summary="out",
            duration_ms=1.0,
        )
        assert stage.details == {}


@pytest.mark.unit
class TestTraceLog:
    def test_creation(self):
        trace = TraceLog(
            query="test query",
            book="my-book",
            embed_model="model-a",
            rerank_model="model-b",
        )
        assert trace.query == "test query"
        assert trace.book == "my-book"
        assert trace.stages == []
        assert trace.total_ms == 0.0

    def test_with_stages(self):
        stage = StageTrace(
            name="embed",
            input_summary="q",
            output_summary="v",
            duration_ms=1.0,
        )
        trace = TraceLog(
            query="q",
            book=None,
            stages=[stage],
            total_ms=1.0,
            embed_model="m",
        )
        assert len(trace.stages) == 1
        assert trace.stages[0].name == "embed"


@pytest.mark.unit
class TestSearchResult:
    def test_with_fragments_only(self):
        frags = [{"text": "hello", "score": 0.9}]
        result = SearchResult(fragments=frags)
        assert result.fragments == frags
        assert result.trace is None

    def test_with_trace(self):
        trace = TraceLog(query="q", book="b", embed_model="m")
        result = SearchResult(fragments=[], trace=trace)
        assert result.trace is not None
        assert result.trace.query == "q"


@pytest.mark.unit
class TestFormatTrace:
    def _make_trace(self, with_rerank=True):
        stages = [
            StageTrace(
                name="embed",
                input_summary='Query: "What is Paris?" (15 chars)',
                output_summary="Model: multilingual-e5-small (384d)",
                duration_ms=2.1,
                details={"model": "multilingual-e5-small", "dimension": 384},
            ),
            StageTrace(
                name="retrieve",
                input_summary="Collection: tiny_sample | Limit: 20",
                output_summary="Retrieved 6 candidates",
                duration_ms=4.3,
                details={
                    "candidates": [
                        {
                            "rank": 1, "page": 1, "score": 0.89,
                            "text_preview": "Paris, the capital of France...",
                        },
                        {
                            "rank": 2, "page": 2, "score": 0.72,
                            "text_preview": "Berlin is the capital...",
                        },
                    ],
                },
            ),
        ]
        if with_rerank:
            stages.append(
                StageTrace(
                    name="rerank",
                    input_summary="Model: ms-marco-MiniLM | 6 candidates",
                    output_summary="Rescored and returned top 8",
                    duration_ms=15.2,
                    details={
                        "rank_changes": [
                            {
                                "page": 1, "before": 1, "after": 1,
                                "delta": 0, "bi_score": 0.89, "ce_score": 12.3,
                            },
                            {
                                "page": 2, "before": 2, "after": 2,
                                "delta": 0, "bi_score": 0.72, "ce_score": 8.1,
                            },
                        ],
                    },
                )
            )

        return TraceLog(
            query="What is Paris?",
            book="tiny_sample",
            stages=stages,
            total_ms=21.6,
            embed_model="multilingual-e5-small",
            rerank_model="ms-marco-MiniLM-L-6-v2" if with_rerank else None,
        )

    def test_contains_query(self):
        output = format_trace(self._make_trace())
        assert '"What is Paris?"' in output

    def test_contains_book(self):
        output = format_trace(self._make_trace())
        assert "tiny_sample" in output

    def test_contains_timing(self):
        output = format_trace(self._make_trace())
        assert "21.6ms" in output

    def test_contains_stage_labels(self):
        output = format_trace(self._make_trace())
        assert "Embedding" in output
        assert "Vector Search" in output
        assert "Cross-Encoder Reranking" in output

    def test_contains_candidates(self):
        output = format_trace(self._make_trace())
        assert "Paris, the capital of France" in output
        assert "Berlin is the capital" in output

    def test_contains_rerank_changes(self):
        output = format_trace(self._make_trace())
        assert "Rerank effect:" in output
        assert "rank 1→1" in output

    def test_rerank_promotes(self):
        trace = self._make_trace(with_rerank=True)
        trace.stages[2].details["rank_changes"] = [
            {"page": 3, "before": 3, "after": 1, "delta": 2, "bi_score": 0.5, "ce_score": 15.0},
        ]
        output = format_trace(trace)
        assert "rank 3→1" in output
        assert "↑2" in output

    def test_rerank_demotes(self):
        trace = self._make_trace(with_rerank=True)
        trace.stages[2].details["rank_changes"] = [
            {"page": 1, "before": 1, "after": 3, "delta": -2, "bi_score": 0.89, "ce_score": 2.0},
        ]
        output = format_trace(trace)
        assert "rank 1→3" in output
        assert "↓2" in output

    def test_no_rerank_model(self):
        output = format_trace(self._make_trace(with_rerank=False))
        assert "Reranker" not in output
        assert "Cross-Encoder" not in output
