import json
import os
import re
import sys
import glob
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import fitz

from src.config import BOOKS_DIR, EXTRACTED_DIR, CHUNKS_FILE, METADATA_FILE
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, collection_name
from src.embeddings import get_model, embed
from src.qdrant_store import get_qdrant_client, ensure_collection, delete_collection, list_collections

CHAPTER_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:Rozdział\s+[IVXLCDM\d]+|Chapter\s+\d+|CZĘŚĆ\s+[IVXLCDM\d]+|Tom\s+\d+)",
    re.IGNORECASE,
)


def extract_pdf(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        pages.append({
            "page_num": i + 1,
            "text": text.strip(),
        })
    doc.close()
    return pages


def detect_chapter(text: str) -> str | None:
    match = re.search(r"(Rozdział\s+[\dIVXLCDM]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(Chapter\s+\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    match = re.search(r"(CZĘŚĆ\s+[\dIVXLCDM]+)", text, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def get_page_boundaries(pages_data: list[dict]) -> list[int]:
    boundaries = []
    total = 0
    for p in pages_data:
        total += len(p["text"]) + 1
        boundaries.append(total)
    return boundaries


def get_full_text_with_page_info(pages_data: list[dict]) -> tuple[str, list[int]]:
    segments = []
    page_nums = []
    for p in pages_data:
        segments.append(p["text"])
        page_nums.append(p["page_num"])
    full_text = "\n".join(segments)
    boundaries = get_page_boundaries(pages_data)
    return full_text, boundaries


def _page_at_position(page_boundaries: list[int], page_nums: list[int], pos: int) -> int:
    for i, boundary in enumerate(page_boundaries):
        if pos < boundary:
            return page_nums[i]
    return page_nums[-1] if page_nums else 1


def chunk_text(text: str, page_boundaries: list[int], page_nums: list[int]) -> list[dict]:
    model = get_model()
    tokenizer = model.tokenizer
    chunks = []

    approx_chars_per_chunk = int(CHUNK_SIZE * 4)
    overlap_chars = int(CHUNK_OVERLAP * 4)
    char_pos = 0

    while char_pos < len(text):
        end_pos = min(char_pos + approx_chars_per_chunk, len(text))
        raw = text[char_pos:end_pos]

        token_count = len(tokenizer.encode(raw))
        while token_count > CHUNK_SIZE and end_pos > char_pos + 50:
            end_pos -= 10
            raw = text[char_pos:end_pos]
            token_count = len(tokenizer.encode(raw))

        raw = raw.strip()
        if not raw:
            char_pos = end_pos
            continue

        start_page = _page_at_position(page_boundaries, page_nums, char_pos)
        end_page = _page_at_position(page_boundaries, page_nums, end_pos)

        chunks.append({
            "text": raw,
            "start_page": start_page,
            "end_page": end_page,
        })

        next_pos = end_pos - overlap_chars
        if next_pos <= char_pos:
            next_pos = end_pos
        char_pos = next_pos

    return chunks


def process_book(pdf_path: str) -> dict:
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"  Processing: {book_name}")

    pages_data = extract_pdf(pdf_path)
    full_text, page_boundaries = get_full_text_with_page_info(pages_data)
    page_nums = [p["page_num"] for p in pages_data]

    print(f"    Pages: {len(pages_data)}, text length: {len(full_text)} chars")

    chunks = chunk_text(full_text, page_boundaries, page_nums)

    result_chunks = []
    current_chapter = None
    for chunk in chunks:
        detected = detect_chapter(chunk["text"])
        if detected:
            current_chapter = detected
        result_chunks.append({
            "text": chunk["text"],
            "book": book_name,
            "chapter": current_chapter or "unknown",
            "start_page": chunk["start_page"],
            "end_page": chunk["end_page"],
        })

    extracted_path = os.path.join(EXTRACTED_DIR, f"{book_name}.txt")
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    with open(extracted_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"    Chunks created: {len(result_chunks)}")
    return {"book": book_name, "chunks": result_chunks, "total_pages": len(pages_data)}


def index_book(pdf_path: str, reindex: bool = False):
    book_name = os.path.splitext(os.path.basename(pdf_path))[0]
    coll = collection_name(book_name)

    qdrant = get_qdrant_client()

    if reindex:
        if coll in list_collections(qdrant):
            delete_collection(coll, qdrant)

    ensure_collection(coll, qdrant)

    result = process_book(pdf_path)
    chunks = result["chunks"]

    if not chunks:
        print(f"  No chunks to index for '{book_name}'.")
        return result

    print(f"  Generating embeddings for {len(chunks)} chunks...")
    texts = [c["text"] for c in chunks]
    vectors = embed(texts)

    print(f"  Storing in Qdrant collection '{coll}'...")
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append({
            "id": i + 1,
            "vector": vector,
            "payload": {
                "text": chunk["text"],
                "book": chunk["book"],
                "chapter": chunk["chapter"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
            },
        })

    qdrant.upsert(
        collection_name=coll,
        points=points,
    )

    print(f"  Done! Indexed {len(chunks)} chunks into '{coll}'.")
    return result


def ingest_all(reindex: str | None = None):
    pdf_files = sorted(glob.glob(os.path.join(BOOKS_DIR, "*.pdf")))
    if not pdf_files:
        print("No PDF files found in books/ directory.")
        return

    qdrant = get_qdrant_client()

    if reindex:
        pdf_path = os.path.join(BOOKS_DIR, f"{reindex}.pdf")
        if not os.path.exists(pdf_path):
            possible = [os.path.splitext(os.path.basename(p))[0] for p in pdf_files]
            print(f"Book '{reindex}' not found. Available: {possible}")
            return
        print(f"Re-indexing: {reindex}")
        index_book(pdf_path, reindex=True)
        return

    all_results = []
    for pdf_path in pdf_files:
        print(f"\nIndexing: {os.path.basename(pdf_path)}")
        result = index_book(pdf_path, reindex=False)
        all_results.append(result)

    total = sum(r["total_pages"] for r in all_results)
    print(f"\nDone! Processed {len(all_results)} books, {total} total pages.")


def delete_book(book_name: str):
    coll = collection_name(book_name)
    qdrant = get_qdrant_client()
    collections = list_collections(qdrant)

    if coll not in collections:
        possible = [c for c in collections if c != "_point_vector"]  # skip internal
        print(f"Collection '{coll}' not found. Available: {possible}")
        return

    delete_collection(coll, qdrant)
    print(f"Book '{book_name}' removed from knowledge base.")


def list_books():
    qdrant = get_qdrant_client()
    collections = list_collections(qdrant)
    if not collections:
        print("No books in the knowledge base.")
        return
    print("Books in knowledge base:")
    for c in sorted(collections):
        count_result = qdrant.count(collection_name=c, exact=True)
        total = count_result.count if hasattr(count_result, 'count') else 0
        print(f"  - {c} ({total} chunks)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PDF-RAG ingestion pipeline")
    parser.add_argument("--reindex", type=str, help="Re-index a specific book by name")
    parser.add_argument("--delete", type=str, help="Delete a book from the knowledge base")
    parser.add_argument("--list", action="store_true", help="List all books in the knowledge base")
    args = parser.parse_args()

    if args.list:
        list_books()
    elif args.delete:
        delete_book(args.delete)
    else:
        ingest_all(reindex=args.reindex)
