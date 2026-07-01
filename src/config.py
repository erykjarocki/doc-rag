import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BOOKS_DIR = os.path.join(BASE_DIR, "books")
EXTRACTED_DIR = os.path.join(BASE_DIR, "data", "extracted")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "chunks", "chunks.json")
METADATA_FILE = os.path.join(BASE_DIR, "data", "metadata", "metadata.json")
QDRANT_PATH = os.path.join(BASE_DIR, "vector_db", "qdrant")

EMBED_MODEL = "intfloat/multilingual-e5-small"
EMBED_DIM = 384

QDRANT_HOST = "localhost"
QDRANT_PORT = 6333

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 8

_COLLECTION_SAFE = re.compile(r"[^a-zA-Z0-9_-]")


def collection_name(book_name: str) -> str:
    safe = _COLLECTION_SAFE.sub("_", book_name).strip("_").lower()
    return safe or "book"
