import time

from src.config import EMBED_MODEL, RERANK_ENABLED, RERANK_MODEL, RERANK_TOP_N, TOP_K
from src.embeddings import embed_query
from src.qdrant_store import get_qdrant_client, list_collections
from src.trace import SearchResult, StageTrace, TraceLog
from src.utils import collection_name


def search_book(
    query: str,
    top_k: int = TOP_K,
    book: str | None = None,
    rerank: bool | None = None,
    trace: bool = False,
) -> list[dict] | SearchResult:
    """Search the vector database for chunks relevant to a query.

    If book is specified, searches only that collection. Otherwise searches
    all collections proportionally and merges results by score.

    Optionally applies cross-encoder re-ranking for higher precision.
    When enabled, retrieves more candidates initially (RERANK_TOP_N),
    then rescores and returns top_k results.

    Args:
        query: Natural language question or search terms.
        top_k: Maximum number of results to return (default: 8).
        book: Optional book name to filter search to a single collection.
        rerank: Whether to apply cross-encoder re-ranking. If None,
            uses RERANK_ENABLED from config.
        trace: If True, returns a SearchResult with pipeline trace data.

    Returns:
        When trace=False: list of dicts with 'text', 'book', 'chapter',
        'start_page', 'end_page', and 'score' keys.
        When trace=True: SearchResult wrapping the list plus a TraceLog.
    """
    use_rerank = rerank if rerank is not None else RERANK_ENABLED
    retrieval_limit = RERANK_TOP_N if use_rerank else top_k

    stages = []
    total_start = time.perf_counter()

    # --- Stage 1: Embedding ---
    t0 = time.perf_counter()
    query_vector = embed_query(query)
    embed_ms = (time.perf_counter() - t0) * 1000

    stages.append(
        StageTrace(
            name="embed",
            input_summary=f'Query: "{query}" ({len(query)} chars)',
            output_summary=f"Model: {EMBED_MODEL} ({len(query_vector)}d)",
            duration_ms=round(embed_ms, 1),
            details={"model": EMBED_MODEL, "dimension": len(query_vector)},
        )
    )

    # --- Stage 2: Vector Search ---
    t0 = time.perf_counter()
    client = get_qdrant_client()

    if book:
        coll = collection_name(book)
        if coll not in list_collections(client):
            results = []
        else:
            resp = client.query_points(
                collection_name=coll,
                query=query_vector,
                limit=retrieval_limit,
            )
            results = _format_results(resp.points)
    else:
        collections = [c for c in list_collections(client)]
        if not collections:
            results = []
        else:
            all_results = []
            per_collection = max(1, retrieval_limit // len(collections)) + 2

            for coll in collections:
                try:
                    resp = client.query_points(
                        collection_name=coll,
                        query=query_vector,
                        limit=per_collection,
                    )
                    all_results.extend(_format_results(resp.points))
                except Exception:
                    continue

            all_results.sort(key=lambda x: x["score"], reverse=True)
            results = all_results[:retrieval_limit]

    retrieve_ms = (time.perf_counter() - t0) * 1000

    candidates_for_trace = [
        {
            "rank": i + 1,
            "page": r.get("start_page", "?"),
            "chapter": r.get("chapter", ""),
            "score": r["score"],
            "text_preview": r["text"][:60] + ("..." if len(r["text"]) > 60 else ""),
        }
        for i, r in enumerate(results)
    ]

    stages.append(
        StageTrace(
            name="retrieve",
            input_summary=f"Collection: {book or 'all'} | Limit: {retrieval_limit}",
            output_summary=f"Retrieved {len(results)} candidates",
            duration_ms=round(retrieve_ms, 1),
            details={
                "collection": book or "all",
                "candidates_count": len(results),
                "retrieval_limit": retrieval_limit,
                "candidates": candidates_for_trace,
            },
        )
    )

    # --- Stage 3: Reranking ---
    rank_changes = None
    if use_rerank and results:
        t0 = time.perf_counter()
        from src.reranker import rerank_with_analysis

        results, rank_changes = rerank_with_analysis(
            query, results, top_k=top_k, model_name=RERANK_MODEL
        )
        rerank_ms = (time.perf_counter() - t0) * 1000

        n_input = len(rank_changes) if rank_changes else 0
        rerank_details: dict = {
            "model": RERANK_MODEL,
            "input_count": n_input + len(results),
        }
        if rank_changes:
            rerank_details["rank_changes"] = [
                {
                    "page": rc["page"],
                    "chapter": rc["chapter"],
                    "before": rc["before"],
                    "after": rc["after"],
                    "delta": rc["delta"],
                    "bi_score": round(rc["bi_score"], 4),
                    "ce_score": round(rc["ce_score"], 4),
                }
                for rc in rank_changes
            ]

        rerank_input = f"Model: {RERANK_MODEL} | {n_input} candidates"
        stages.append(
            StageTrace(
                name="rerank",
                input_summary=rerank_input,
                output_summary=f"Rescored and returned top {len(results)}",
                duration_ms=round(rerank_ms, 1),
                details=rerank_details,
            )
        )
    elif use_rerank:
        stages.append(
            StageTrace(
                name="rerank",
                input_summary=f"Model: {RERANK_MODEL}",
                output_summary="No candidates to rerank",
                duration_ms=0.0,
                details={"model": RERANK_MODEL, "input_count": 0},
            )
        )

    total_ms = (time.perf_counter() - total_start) * 1000
    final = results[:top_k]

    if not trace:
        return final

    trace_log = TraceLog(
        query=query,
        book=book,
        stages=stages,
        total_ms=round(total_ms, 1),
        embed_model=EMBED_MODEL,
        rerank_model=RERANK_MODEL if use_rerank else None,
    )
    return SearchResult(fragments=final, trace=trace_log)


def _format_results(points) -> list[dict]:
    """Convert Qdrant result points into a standardized dict format.

    Args:
        points: List of Qdrant ScoredPoint objects.

    Returns:
        List of dicts with text, book, chapter, pages, and score.
    """
    fragments = []
    for hit in points:
        p = hit.payload
        fragments.append(
            {
                "text": p["text"],
                "book": p["book"],
                "chapter": p.get("chapter", ""),
                "start_page": p.get("start_page", ""),
                "end_page": p.get("end_page", ""),
                "source_file": p.get("source_file", ""),
                "score": round(hit.score, 4),
            }
        )
    return fragments


def format_fragments_for_prompt(fragments: list[dict]) -> str:
    """Format search results as numbered text blocks with Polish citations.

    Args:
        fragments: List of fragment dicts from search_book().

    Returns:
        Formatted string with numbered blocks and source citations
        (e.g. "[1] text... Źródło: book, chapter, str. X-Y").
    """
    lines = []
    for i, f in enumerate(fragments, 1):
        source = f"Źródło: {f['book']}"
        if f.get("chapter"):
            source += f", {f['chapter']}"
        if f.get("start_page"):
            source += f", str. {f['start_page']}"
            if f.get("end_page") and f["end_page"] != f["start_page"]:
                source += f"-{f['end_page']}"

        lines.append(f"[{i}] {f['text']}\n\n{source}\n---")
    return "\n".join(lines)
