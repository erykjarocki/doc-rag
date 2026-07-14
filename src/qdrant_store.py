from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams

from src.config import EMBED_DIM, QDRANT_HOST, QDRANT_PORT
from src.log import get_logger

logger = get_logger(__name__)

_client = None


def get_qdrant_client() -> QdrantClient:
    """Get or create a Qdrant client connected to the Docker instance.

    Returns:
        QdrantClient connected to localhost:6333.
    """
    global _client
    if _client is None:
        _client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    return _client


def ensure_collection(name: str, client: QdrantClient | None = None):
    """Create a Qdrant collection if it doesn't already exist.

    Creates a collection with EMBED_DIM-dimensional cosine vectors
    and a keyword index on the "book" field for filtered searches.

    Args:
        name: Collection name (use collection_name() to sanitize).
        client: Optional QdrantClient instance (uses default if None).
    """
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
        logger.info("Created collection '%s'", name, extra={"collection": name, "action": "create"})
    else:
        check_dimension_mismatch(name, client)
        logger.debug("Collection '%s' already exists", name, extra={"collection": name})


def delete_collection(name: str, client: QdrantClient | None = None):
    """Delete a Qdrant collection and all its vectors.

    Args:
        name: Collection name to delete.
        client: Optional QdrantClient instance (uses default if None).
    """
    if client is None:
        client = get_qdrant_client()
    client.delete_collection(collection_name=name)
    logger.info("Deleted collection '%s'", name, extra={"collection": name, "action": "delete"})


def list_collections(client: QdrantClient | None = None) -> list[str]:
    """List all collection names in Qdrant.

    Args:
        client: Optional QdrantClient instance (uses default if None).

    Returns:
        List of collection name strings.
    """
    if client is None:
        client = get_qdrant_client()
    return [c.name for c in client.get_collections().collections]


def check_dimension_mismatch(
    name: str,
    client: QdrantClient | None = None,
    *,
    raise_on_mismatch: bool = False,
) -> bool:
    """Check if a collection's vector dimension matches the configured EMBED_DIM.

    Args:
        name: Collection name to check.
        client: Optional QdrantClient instance (uses default if None).
        raise_on_mismatch: If True, raises ValueError on mismatch.

    Returns:
        True if dimensions match, False if mismatch detected.
    """
    if client is None:
        client = get_qdrant_client()

    try:
        info = client.get_collection(collection_name=name)
        collection_dim = info.config.params.vectors.size
    except Exception:
        logger.warning("Could not get collection info for '%s'", name)
        return True  # Can't check, assume OK

    if collection_dim != EMBED_DIM:
        msg = (
            f"Collection '{name}' has dimension {collection_dim} but config "
            f"expects {EMBED_DIM}. The collection was indexed with a different "
            f"model. Re-index with --reindex or update EMBED_DIM in config."
        )
        if raise_on_mismatch:
            raise ValueError(msg)
        logger.warning("Dimension mismatch", extra={
            "collection": name,
            "collection_dim": collection_dim,
            "expected_dim": EMBED_DIM,
        })
        return False

    return True
