from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from src.config import QDRANT_HOST, QDRANT_PORT, EMBED_DIM

_client = None


def get_qdrant_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_collection(name: str, client: QdrantClient | None = None):
    if client is None:
        client = get_qdrant_client()

    collections = client.get_collections().collections
    exists = any(c.name == name for c in collections)

    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
        )
        client.create_payload_index(
            collection_name=name,
            field_name="book",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        print(f"  Created collection '{name}'")
    else:
        print(f"  Collection '{name}' already exists")


def delete_collection(name: str, client: QdrantClient | None = None):
    if client is None:
        client = get_qdrant_client()
    client.delete_collection(collection_name=name)
    print(f"  Deleted collection '{name}'")


def list_collections(client: QdrantClient | None = None) -> list[str]:
    if client is None:
        client = get_qdrant_client()
    return [c.name for c in client.get_collections().collections]
