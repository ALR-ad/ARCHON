import pytest
from unittest.mock import patch, MagicMock

from indexing.embeddings.embedder import (
    embed_text,
    set_custom_embedder,
    deterministic_mock_embed,
    OllamaServiceError,
    _EMBEDDING_CACHE,
)
from shared.config import EMBEDDING_DIM
import shared.interfaces as shared_interfaces


def test_embed_text_with_mock():
    set_custom_embedder(deterministic_mock_embed)
    try:
        vec = embed_text("def test(): pass")
        assert isinstance(vec, list)
        assert len(vec) == EMBEDDING_DIM
        assert all(isinstance(x, float) for x in vec)

        # Same text produces identical vector
        vec2 = embed_text("def test(): pass")
        assert vec == vec2

        # Cross-boundary interface check
        vec3 = shared_interfaces.embed_text("def test(): pass")
        assert vec3 == vec
    finally:
        set_custom_embedder(None)


def test_embed_text_caching():
    set_custom_embedder(None)
    fake_vec = [0.42] * EMBEDDING_DIM
    text = "unique_cached_sample_code_block"

    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.embeddings.return_value = {"embedding": fake_vec}
        mock_client_cls.return_value = mock_instance

        # First call hits mock
        res1 = embed_text(text)
        assert res1 == fake_vec
        assert mock_instance.embeddings.call_count == 1

        # Second call hits cache, does not call client again
        res2 = embed_text(text)
        assert res2 == fake_vec
        assert mock_instance.embeddings.call_count == 1


def test_embed_text_ollama_unavailable_raises_clear_error():
    set_custom_embedder(None)
    with patch("ollama.Client") as mock_client_cls:
        mock_instance = MagicMock()
        mock_instance.embeddings.side_effect = ConnectionError("Connection refused")
        mock_client_cls.return_value = mock_instance

        with pytest.raises(OllamaServiceError) as exc_info:
            embed_text("def err(): pass", host="http://invalid-host:11434")

        assert "Ollama embedding service error" in str(exc_info.value)
        assert "ollama pull" in str(exc_info.value)


def test_embed_text_live_ollama_smoke():
    """
    Live smoke test against real local Ollama instance with nomic-embed-text.
    Verifies 768-dim vector generation.
    """
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1.0)
    except Exception:
        pytest.skip("Ollama is not running locally on http://localhost:11434")

    set_custom_embedder(None)
    try:
        vec = embed_text("def smoke_test_function(): return 42")
        assert isinstance(vec, list)
        assert len(vec) == 768
        assert all(isinstance(x, float) for x in vec)
    except OllamaServiceError as e:
        pytest.skip(f"Ollama running but model failed: {e}")

