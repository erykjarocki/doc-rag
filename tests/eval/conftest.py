import os
import sys

import fitz
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

import src.config as config
import src.ingest as ingest
import src.qdrant_store as qdrant_store

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.embeddings import embed
from src.ingest import process_book
from src.qdrant_store import ensure_collection

BOOK_NAME = "tiny_sample"
COLLECTION_NAME = "tiny_sample"


@pytest.fixture(scope="module")
def tiny_pdf(tmp_path_factory):
    """Generate a 2-page PDF with headings and paragraphs about France and Germany."""
    tmp_dir = tmp_path_factory.mktemp("pdfs")
    pdf_path = str(tmp_dir / "tiny_sample.pdf")

    doc = fitz.open()

    page = doc.new_page()
    text = (
        "Chapter 1: France\n\n"
        "Paris is the capital and most populous city of France. The city is "
        "known for its iconic Eiffel Tower, which was built in 1889 for the "
        "World's Fair. Paris attracts millions of tourists every year.\n\n"
        "The Louvre Museum in Paris is the world's largest art museum. It "
        "houses the Mona Lisa painting by Leonardo da Vinci. The museum "
        "receives over nine million visitors annually."
    )
    page.insert_text((72, 72), text, fontsize=11)

    page = doc.new_page()
    text = (
        "Chapter 2: Germany\n\n"
        "Berlin is the capital and largest city of Germany. The city is known "
        "for the Brandenburg Gate, a neoclassical monument built in the 18th "
        "century. Berlin has a rich and complex history.\n\n"
        "The Berlin Wall divided the city from 1961 to 1989. Its fall "
        "symbolized the end of the Cold War. Today Berlin is a major European "
        "cultural center."
    )
    page.insert_text((72, 72), text, fontsize=11)

    doc.save(pdf_path)
    doc.close()
    return pdf_path


@pytest.fixture(scope="module")
def indexed_qdrant(tiny_pdf, tmp_path_factory):
    """Run the full pipeline: extract -> chunk -> detect -> embed -> store.

    Uses real embedding model, real extraction, in-memory Qdrant.
    Returns (qdrant_client, collection_name, chunks).
    """
    tmp_dir = tmp_path_factory.mktemp("data")

    in_memory_client = QdrantClient(":memory:")
    original_client = qdrant_store._client

    qdrant_store._client = in_memory_client
    original_extracted = config.EXTRACTED_DIR
    config.EXTRACTED_DIR = str(tmp_dir / "extracted")
    ingest.EXTRACTED_DIR = config.EXTRACTED_DIR

    try:
        ensure_collection(COLLECTION_NAME, in_memory_client)

        result = process_book(tiny_pdf)
        chunks = result["chunks"]
        assert len(chunks) > 0, "Expected at least one chunk from the test PDF"

        texts = [c["text"] for c in chunks]
        vectors = embed(texts)

        points = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            points.append(
                PointStruct(
                    id=i + 1,
                    vector=vector,
                    payload={
                        "text": chunk["text"],
                        "book": chunk["book"],
                        "chapter": chunk["chapter"],
                        "start_page": chunk["start_page"],
                        "end_page": chunk["end_page"],
                    },
                )
            )

        in_memory_client.upsert(collection_name=COLLECTION_NAME, points=points)
    except Exception:
        qdrant_store._client = original_client
        config.EXTRACTED_DIR = original_extracted
        ingest.EXTRACTED_DIR = original_extracted
        raise

    yield in_memory_client, COLLECTION_NAME, chunks

    qdrant_store._client = original_client
    config.EXTRACTED_DIR = original_extracted
    ingest.EXTRACTED_DIR = original_extracted
    try:
        in_memory_client.delete_collection(collection_name=COLLECTION_NAME)
    except Exception:
        pass
