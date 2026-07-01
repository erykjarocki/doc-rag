import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.retriever import search_book, format_fragments_for_prompt
from src.qdrant_store import list_collections, delete_collection, get_qdrant_client

app = FastAPI(title="PDF-RAG API", version="1.1.0")


class QueryRequest(BaseModel):
    question: str
    book: str | None = None


class Fragment(BaseModel):
    text: str
    book: str
    chapter: str
    start_page: str | int
    end_page: str | int


class QueryResponse(BaseModel):
    context: list[Fragment]
    formatted: str


@app.post("/query")
def query(req: QueryRequest):
    fragments = search_book(req.question, book=req.book)
    return QueryResponse(
        context=[
            Fragment(
                text=f["text"],
                book=f["book"],
                chapter=f.get("chapter", ""),
                start_page=f.get("start_page", ""),
                end_page=f.get("end_page", ""),
            )
            for f in fragments
        ],
        formatted=format_fragments_for_prompt(fragments),
    )


@app.get("/books")
def list_books():
    return {"books": list_collections()}


@app.delete("/books/{name}")
def remove_book(name: str):
    client = get_qdrant_client()
    collections = list_collections(client)
    if name not in collections:
        raise HTTPException(status_code=404, detail=f"Book '{name}' not found")
    delete_collection(name, client)
    return {"status": "deleted", "book": name}


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
