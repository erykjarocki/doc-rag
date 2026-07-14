import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from src.config import (
    CONFIG_FILE,
    Settings,
    _apply_env_overrides,
    generate_config,
    load_config,
)


@pytest.mark.unit
class TestConfigDefaults:
    def test_default_settings(self):
        s = Settings()
        assert s.embedding.model == "intfloat/multilingual-e5-base"
        assert s.embedding.dimension == 768
        assert s.qdrant.host == "localhost"
        assert s.qdrant.port == 6333
        assert s.chunking.size == 384
        assert s.chunking.overlap == 50
        assert s.search.top_k == 8
        assert s.api.host == "0.0.0.0"
        assert s.api.port == 8000

    def test_generate_config_returns_dict(self):
        cfg = generate_config()
        assert isinstance(cfg, dict)
        assert "embedding" in cfg
        assert "qdrant" in cfg
        assert "chunking" in cfg
        assert "search" in cfg
        assert "api" in cfg


@pytest.mark.unit
class TestEnvOverrides:
    def test_override_embedding_model(self):
        with patch.dict(os.environ, {"EMBED_MODEL": "custom/model"}):
            s = _apply_env_overrides(Settings())
            assert s.embedding.model == "custom/model"

    def test_override_qdrant_port(self):
        with patch.dict(os.environ, {"QDRANT_PORT": "9999"}):
            s = _apply_env_overrides(Settings())
            assert s.qdrant.port == 9999

    def test_override_chunk_size(self):
        with patch.dict(os.environ, {"CHUNK_SIZE": "512"}):
            s = _apply_env_overrides(Settings())
            assert s.chunking.size == 512

    def test_override_top_k(self):
        with patch.dict(os.environ, {"TOP_K": "15"}):
            s = _apply_env_overrides(Settings())
            assert s.search.top_k == 15

    def test_no_override_when_env_missing(self):
        env = os.environ.copy()
        env.pop("EMBED_MODEL", None)
        with patch.dict(os.environ, env, clear=True):
            s = _apply_env_overrides(Settings())
            assert s.embedding.model == "intfloat/multilingual-e5-base"


@pytest.mark.unit
class TestLoadConfigFromFile:
    def test_load_from_file(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({
            "embedding": {"model": "custom/model", "dimension": 768},
            "qdrant": {"host": "remote-host", "port": 6334},
        }))
        with patch("src.config.CONFIG_FILE", config_file):
            s = load_config()
            assert s.embedding.model == "custom/model"
            assert s.embedding.dimension == 768
            assert s.qdrant.host == "remote-host"
            assert s.qdrant.port == 6334
            # Defaults preserved for untouched sections
            assert s.chunking.size == 384

    def test_missing_file_uses_defaults(self):
        with patch("src.config.CONFIG_FILE", Path("/nonexistent/config.json")):
            s = load_config()
            assert s.embedding.model == "intfloat/multilingual-e5-base"

    def test_invalid_json_uses_defaults(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text("not json {{{")
        with patch("src.config.CONFIG_FILE", config_file):
            s = load_config()
            assert s.embedding.model == "intfloat/multilingual-e5-base"

    def test_partial_config_preserves_defaults(self, tmp_path):
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps({"search": {"top_k": 20}}))
        with patch("src.config.CONFIG_FILE", config_file):
            s = load_config()
            assert s.search.top_k == 20
            assert s.embedding.model == "intfloat/multilingual-e5-base"
            assert s.qdrant.port == 6333


@pytest.mark.unit
class TestConfigFileConstants:
    def test_config_file_path(self):
        assert CONFIG_FILE == Path.home() / ".config" / "doc-rag" / "config.json"
