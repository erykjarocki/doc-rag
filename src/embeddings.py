from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL
from src.log import get_logger

logger = get_logger(__name__)

_model = None
_requires_prefix = None


def get_model():
    """Load and cache the SentenceTransformer embedding model (singleton).

    Returns:
        Loaded SentenceTransformer model with max_seq_length set.
    """
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", EMBED_MODEL, extra={"model": EMBED_MODEL})
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
        if _model.max_seq_length is None:
            _model.max_seq_length = 512
        logger.info("Embedding model loaded", extra={"model": EMBED_MODEL})
    return _model


def get_tokenizer():
    """Return the tokenizer from the loaded embedding model.

    Returns:
        Tokenizer instance used by the SentenceTransformer model.
    """
    return get_model().tokenizer


def _check_prefixes():
    """Check if the embedding model requires passage/query prefixes (E5 models).

    Returns:
        True if the model name contains "e5", indicating prefix requirement.
    """
    global _requires_prefix
    if _requires_prefix is None:
        name = EMBED_MODEL.lower()
        _requires_prefix = "e5" in name or "e5-" in name
    return _requires_prefix


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of text chunks into vectors (for indexing).

    Prepends "passage: " prefix for E5 models. Returns normalized vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of normalized vectors (one per input text).
    """
    model = get_model()
    if _check_prefixes():
        texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single query string into a vector (for searching).

    Prepends "query: " prefix for E5 models. Returns normalized vector.

    Args:
        text: Query string to embed.

    Returns:
        Normalized vector representing the query.
    """
    model = get_model()
    if _check_prefixes():
        text = f"query: {text}"
    return model.encode([text], normalize_embeddings=True).tolist()[0]
