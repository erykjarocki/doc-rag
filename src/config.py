"""Configuration loader for doc-rag.

Priority: env vars > ~/.config/doc-rag/config.json > defaults.

Usage:
    from src.config import settings
    print(settings.qdrant.host)
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config file location
CONFIG_DIR = Path.home() / ".config" / "doc-rag"
CONFIG_FILE = CONFIG_DIR / "config.json"


@dataclass
class EmbeddingConfig:
    model: str = "intfloat/multilingual-e5-base"
    dimension: int = 768


@dataclass
class QdrantConfig:
    host: str = "localhost"
    port: int = 6333


@dataclass
class ChunkingConfig:
    size: int = 384
    overlap: int = 50


@dataclass
class SearchConfig:
    top_k: int = 8


@dataclass
class ApiConfig:
    host: str = "0.0.0.0"
    port: int = 8000


@dataclass
class RerankConfig:
    enabled: bool = True
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    top_n: int = 20  # Retrieve this many candidates before reranking


@dataclass
class Settings:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    api: ApiConfig = field(default_factory=ApiConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)


def _apply_env_overrides(settings: Settings) -> Settings:
    """Override config values with environment variables if set."""
    env_map = {
        "EMBED_MODEL": (settings.embedding, "model"),
        "EMBED_DIM": (settings.embedding, "dimension"),
        "QDRANT_HOST": (settings.qdrant, "host"),
        "QDRANT_PORT": (settings.qdrant, "port"),
        "CHUNK_SIZE": (settings.chunking, "size"),
        "CHUNK_OVERLAP": (settings.chunking, "overlap"),
        "TOP_K": (settings.search, "top_k"),
        "API_HOST": (settings.api, "host"),
        "API_PORT": (settings.api, "port"),
        "RERANK_ENABLED": (settings.rerank, "enabled"),
        "RERANK_MODEL": (settings.rerank, "model"),
        "RERANK_TOP_N": (settings.rerank, "top_n"),
    }
    for env_key, (obj, attr) in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            current = getattr(obj, attr)
            setattr(obj, attr, type(current)(val))
    return settings


def load_config() -> Settings:
    """Load config from file, apply env overrides, return Settings."""
    settings = Settings()

    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                data = json.load(f)
            if "embedding" in data:
                for k, v in data["embedding"].items():
                    if hasattr(settings.embedding, k):
                        setattr(settings.embedding, k, v)
            if "qdrant" in data:
                for k, v in data["qdrant"].items():
                    if hasattr(settings.qdrant, k):
                        setattr(settings.qdrant, k, v)
            if "chunking" in data:
                for k, v in data["chunking"].items():
                    if hasattr(settings.chunking, k):
                        setattr(settings.chunking, k, v)
            if "search" in data:
                for k, v in data["search"].items():
                    if hasattr(settings.search, k):
                        setattr(settings.search, k, v)
            if "api" in data:
                for k, v in data["api"].items():
                    if hasattr(settings.api, k):
                        setattr(settings.api, k, v)
            if "rerank" in data:
                for k, v in data["rerank"].items():
                    if hasattr(settings.rerank, k):
                        setattr(settings.rerank, k, v)
        except (json.JSONDecodeError, OSError):
            pass  # Fall back to defaults

    return _apply_env_overrides(settings)


def generate_config() -> dict:
    """Generate default config as a dict (for writing to file)."""
    return asdict(Settings())


# Singleton
settings = load_config()

# Backward-compatible module-level constants
EXTRACTED_DIR = os.path.join(BASE_DIR, "data", "extracted")
CHUNKS_FILE = os.path.join(BASE_DIR, "data", "chunks", "chunks.json")
METADATA_FILE = os.path.join(BASE_DIR, "data", "metadata", "metadata.json")
QDRANT_PATH = os.path.join(BASE_DIR, "vector_db", "qdrant")

EMBED_MODEL = settings.embedding.model
EMBED_DIM = settings.embedding.dimension

QDRANT_HOST = settings.qdrant.host
QDRANT_PORT = settings.qdrant.port

CHUNK_SIZE = settings.chunking.size
CHUNK_OVERLAP = settings.chunking.overlap
TOP_K = settings.search.top_k

API_HOST = settings.api.host
API_PORT = settings.api.port

RERANK_ENABLED = settings.rerank.enabled
RERANK_MODEL = settings.rerank.model
RERANK_TOP_N = settings.rerank.top_n
