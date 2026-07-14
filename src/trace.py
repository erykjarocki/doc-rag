"""Structured retrieval tracing for the RAG pipeline.

Captures every stage of a search query — embedding, vector search,
and optional cross-encoder reranking — with timing and metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageTrace:
    """Result of one pipeline stage (embed, retrieve, or rerank)."""

    name: str
    input_summary: str
    output_summary: str
    duration_ms: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class TraceLog:
    """Full trace of a search_book() call through the pipeline."""

    query: str
    book: str | None
    stages: list[StageTrace] = field(default_factory=list)
    total_ms: float = 0.0
    embed_model: str = ""
    rerank_model: str | None = None


@dataclass
class SearchResult:
    """Return type for search_book(trace=True).

    Wraps the fragment list together with the pipeline trace so callers
    can inspect both the results and how they were produced.
    """

    fragments: list[dict]
    trace: TraceLog | None = None


def format_trace(trace: TraceLog) -> str:
    """Render a TraceLog as human-readable text.

    Produces a multi-section report showing each pipeline stage's
    timing, metadata, and (for reranking) the before/after ranking.
    """
    lines = [
        f'Query: "{trace.query}"',
        f"Book: {trace.book or 'all'} | Embed model: {trace.embed_model}",
    ]
    if trace.rerank_model:
        lines[-1] += f" | Reranker: {trace.rerank_model}"
    lines.append("")

    for i, stage in enumerate(trace.stages, 1):
        label = _stage_label(stage.name)
        lines.append(f"STAGE {i}: {label} ({stage.duration_ms:.1f}ms)")
        lines.append(f"  {stage.input_summary}")
        lines.append(f"  {stage.output_summary}")

        if stage.name == "retrieve" and "candidates" in stage.details:
            lines.append("")
            lines.append("  Retrieved candidates:")
            for c in stage.details["candidates"]:
                lines.append(
                    f"    #{c['rank']}  page={c['page']}  "
                    f"score={c['score']:.4f}  "
                    f'"{c["text_preview"]}"'
                )

        if stage.name == "rerank" and "rank_changes" in stage.details:
            lines.append("")
            lines.append("  Rerank effect:")
            for r in stage.details["rank_changes"]:
                direction = "unchanged" if r["delta"] == 0 else (
                    f"↑{r['delta']}" if r["delta"] > 0 else f"↓{abs(r['delta'])}"
                )
                lines.append(
                    f"    page={r['page']}  "
                    f"rank {r['before']}→{r['after']}  "
                    f"bi={r['bi_score']:.4f} ce={r['ce_score']:.4f}  "
                    f"({direction})"
                )

        lines.append("")

    lines.append(f"Total: {trace.total_ms:.1f}ms")
    return "\n".join(lines)


def _stage_label(name: str) -> str:
    labels = {
        "embed": "Embedding",
        "retrieve": "Vector Search",
        "rerank": "Cross-Encoder Reranking",
    }
    return labels.get(name, name)
