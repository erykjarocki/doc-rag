"""Cross-encoder re-ranking for improved retrieval precision.

After initial retrieval with dense embeddings (fast but imprecise),
a cross-encoder rescores each (query, document) pair for much higher
accuracy. This is the "Advanced RAG" re-ranking stage.
"""

from __future__ import annotations

from typing import Optional

from src.log import get_logger

logger = get_logger(__name__)

_reranker = None
_model_name: Optional[str] = None


def get_reranker(model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load and cache the cross-encoder model (singleton).

    The cross-encoder processes (query, document) pairs jointly, producing
    a relevance score. Unlike bi-encoders (used for initial retrieval),
    cross-encoders see both query and document simultaneously, enabling
    much more accurate relevance judgment at the cost of speed.

    Args:
        model_name: HuggingFace model identifier. Default is
            cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality).

    Returns:
        Loaded CrossEncoder model instance.
    """
    global _reranker, _model_name

    if _reranker is None or _model_name != model_name:
        from sentence_transformers import CrossEncoder

        logger.info("Loading cross-encoder model: %s", model_name, extra={"model": model_name})
        _reranker = CrossEncoder(model_name, max_length=512)
        _model_name = model_name
        logger.info("Cross-encoder loaded", extra={"model": model_name})

    return _reranker


def rerank(
    query: str,
    fragments: list[dict],
    top_k: int = 5,
    model_name: Optional[str] = None,
) -> list[dict]:
    """Re-rank retrieval results using a cross-encoder model.

    Takes the top-N results from initial dense retrieval and rescores them
    using a cross-encoder that sees both query and document jointly. This
    produces more accurate relevance scores but is slower than bi-encoder
    search, so it's applied only to the top candidates.

    Args:
        query: The original search query.
        fragments: List of fragment dicts from initial retrieval (must have
            'text' key). Typically top-10 to top-20 results.
        top_k: Number of top results to return after re-ranking.
        model_name: Optional model override. Uses default if None.

    Returns:
        Re-ranked list of fragments with added 'rerank_score' key,
        sorted by relevance, limited to top_k results.
    """
    if not fragments:
        return []

    model = get_reranker(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(query, f["text"]) for f in fragments]
    scores = model.predict(pairs)

    for frag, score in zip(fragments, scores):
        frag["rerank_score"] = round(float(score), 4)

    fragments.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
    return fragments[:top_k]


def rerank_with_analysis(
    query: str,
    fragments: list[dict],
    top_k: int = 5,
    model_name: str | None = None,
) -> tuple[list[dict], list[dict]]:
    """Re-rank and return rank change analysis alongside results.

    Same as rerank() but also computes before/after rank positions for
    each fragment, enabling the trace system to show how the cross-encoder
    reordered the bi-encoder results.

    Returns:
        Tuple of (reranked_fragments, rank_changes) where rank_changes
        is a list of dicts with keys: page, before, after, delta,
        bi_score, ce_score.
    """
    if not fragments:
        return [], []

    model = get_reranker(model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2")

    pairs = [(query, f["text"]) for f in fragments]
    scores = model.predict(pairs)

    for frag, score in zip(fragments, scores):
        frag["rerank_score"] = round(float(score), 4)

    # Assign stable IDs before sorting so we can match after reorder
    for idx, frag in enumerate(fragments):
        frag["_rerank_idx"] = idx

    # Build rank analysis from original order
    rank_changes = []
    for i, frag in enumerate(fragments):
        rank_changes.append(
            {
                "_rerank_idx": frag["_rerank_idx"],
                "page": frag.get("start_page", "?"),
                "chapter": frag.get("chapter", ""),
                "before": i + 1,
                "bi_score": frag["score"],
            }
        )

    fragments.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

    # Fill in after-rank and delta using stable IDs
    for i, frag in enumerate(fragments):
        idx = frag["_rerank_idx"]
        rc = rank_changes[idx]
        rc["after"] = i + 1
        rc["ce_score"] = frag["rerank_score"]
        rc["delta"] = rc["before"] - rc["after"]

    # Clean up internal key
    for frag in fragments:
        frag.pop("_rerank_idx", None)
    for rc in rank_changes:
        rc.pop("_rerank_idx", None)

    return fragments[:top_k], rank_changes


def is_reranker_available() -> bool:
    """Check if the reranker model is loaded and available."""
    return _reranker is not None
