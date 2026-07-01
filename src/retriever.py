from src.config import TOP_K, collection_name
from src.embeddings import embed_query
from src.qdrant_store import get_qdrant_client, list_collections


def search_book(query: str, top_k: int = TOP_K, book: str | None = None) -> list[dict]:
    query_vector = embed_query(query)
    client = get_qdrant_client()

    if book:
        coll = collection_name(book)
        if coll not in list_collections(client):
            return []
        resp = client.query_points(
            collection_name=coll,
            query=query_vector,
            limit=top_k,
        )
        return _format_results(resp.points)

    collections = [c for c in list_collections(client)]
    if not collections:
        return []

    all_results = []
    per_collection = max(1, top_k // len(collections)) + 2

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
    return all_results[:top_k]


def _format_results(points) -> list[dict]:
    fragments = []
    for hit in points:
        p = hit.payload
        fragments.append({
            "text": p["text"],
            "book": p["book"],
            "chapter": p.get("chapter", ""),
            "start_page": p.get("start_page", ""),
            "end_page": p.get("end_page", ""),
            "score": round(hit.score, 4),
        })
    return fragments


def format_fragments_for_prompt(fragments: list[dict]) -> str:
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
