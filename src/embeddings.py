from sentence_transformers import SentenceTransformer

from src.config import EMBED_MODEL

_model = None
_requires_prefix = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL, trust_remote_code=True)
    return _model


def _check_prefixes():
    global _requires_prefix
    if _requires_prefix is None:
        name = EMBED_MODEL.lower()
        _requires_prefix = "e5" in name or "e5-" in name
    return _requires_prefix


def embed(texts: list[str]) -> list[list[float]]:
    model = get_model()
    if _check_prefixes():
        texts = [f"passage: {t}" for t in texts]
    return model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()


def embed_query(text: str) -> list[float]:
    model = get_model()
    if _check_prefixes():
        text = f"query: {text}"
    return model.encode([text], normalize_embeddings=True).tolist()[0]
