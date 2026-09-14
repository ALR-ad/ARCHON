"""
indexing/embeddings/embedder.py

PERSON A: implements embed_text() wrapping a local Ollama call to
shared.config.EMBEDDING_MODEL. Includes content-hash caching, dimension
validation, and clear error messages when Ollama is unavailable.
"""

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import ollama
from shared.config import EMBEDDING_DIM, EMBEDDING_MODEL
from indexing.config import OLLAMA_HOST

logger = logging.getLogger(__name__)

# Global in-memory cache: sha256(text) -> embedding vector
_EMBEDDING_CACHE: Dict[str, List[float]] = {}
_CACHE_FILE_PATH: Optional[Path] = None

# Optional test hook for offline mocking
_CUSTOM_EMBEDDER: Optional[Callable[[str], List[float]]] = None


class OllamaServiceError(RuntimeError):
    """Raised when Ollama is unreachable or model is unavailable."""
    pass


def set_custom_embedder(fn: Optional[Callable[[str], List[float]]]) -> None:
    """Set a custom embedding function (primarily for unit tests and CI)."""
    global _CUSTOM_EMBEDDER
    _CUSTOM_EMBEDDER = fn


def init_cache(cache_dir: Optional[str] = None) -> None:
    """Load persistent embedding cache from disk if available."""
    global _EMBEDDING_CACHE, _CACHE_FILE_PATH
    if cache_dir:
        cache_path = Path(cache_dir) / "embedding_cache.json"
        _CACHE_FILE_PATH = cache_path
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    _EMBEDDING_CACHE.update(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load embedding cache from {cache_path}: {e}")


def save_cache() -> None:
    """Persist in-memory embedding cache to disk if cache path is configured."""
    global _EMBEDDING_CACHE, _CACHE_FILE_PATH
    if _CACHE_FILE_PATH:
        try:
            _CACHE_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(_CACHE_FILE_PATH, "w", encoding="utf-8") as f:
                json.dump(_EMBEDDING_CACHE, f)
        except Exception as e:
            logger.warning(f"Failed to save embedding cache: {e}")


def deterministic_mock_embed(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    """
    Generate a deterministic token-aware normalized pseudo-embedding for testing.
    Shared words yield high cosine similarity, while unrelated text yields near-zero similarity.
    """
    import re
    words = re.findall(r"\w+", text.lower())
    if not words:
        words = [text.lower().strip()] if text.strip() else ["empty"]

    vec = [0.0] * dim
    for word in words:
        h = hashlib.sha256(word.encode("utf-8")).digest()
        for i in range(dim):
            b = h[i % len(h)]
            vec[i] += (float(b) / 255.0) * 2.0 - 1.0

    # Normalize to unit length for cosine similarity
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def embed_text(
    text: str,
    host: Optional[str] = None,
    model: Optional[str] = None,
) -> List[float]:
    """
    Generate an embedding vector for a single string.
    Checks content-hash cache before querying Ollama.

    Input:
        text: str -- raw code or markdown text
        host: Optional[str] -- Ollama host (defaults to shared.config.OLLAMA_HOST)
        model: Optional[str] -- Ollama model name (defaults to shared.config.EMBEDDING_MODEL)

    Output:
        List[float] -- vector of dimension shared.config.EMBEDDING_DIM
    """
    if not isinstance(text, str):
        text = str(text)

    # 1. Check custom test override first
    if _CUSTOM_EMBEDDER is not None:
        return _CUSTOM_EMBEDDER(text)

    # 2. Check content-hash cache
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if text_hash in _EMBEDDING_CACHE:
        return _EMBEDDING_CACHE[text_hash]

    target_host = host or OLLAMA_HOST
    target_model = model or EMBEDDING_MODEL

    try:
        client = ollama.Client(host=target_host)
        res = client.embeddings(model=target_model, prompt=text)
        embedding = res.get("embedding")
        if not embedding or not isinstance(embedding, list):
            raise ValueError(f"Ollama returned unexpected embedding shape: {res}")

        # Dimension validation
        if len(embedding) != EMBEDDING_DIM:
            logger.warning(
                f"Embedding dimension mismatch: expected {EMBEDDING_DIM}, got {len(embedding)}. "
                f"Verify EMBEDDING_DIM in shared/config.py matches '{target_model}'."
            )

        # Store in cache
        _EMBEDDING_CACHE[text_hash] = embedding
        return embedding

    except Exception as e:
        msg = (
            f"Ollama embedding service error: Failed to connect to Ollama at '{target_host}' "
            f"or embed using model '{target_model}'.\n"
            f"Please ensure Ollama is installed, running ('ollama serve'), and the model is pulled:\n"
            f"    ollama pull {target_model}\n"
            f"Underlying error: {type(e).__name__}: {e}"
        )
        logger.error(msg)
        raise OllamaServiceError(msg) from e


def embed_batch(
    texts: List[str],
    host: Optional[str] = None,
    model: Optional[str] = None,
) -> List[List[float]]:
    """
    Embed multiple texts, utilizing the content cache for unchanged texts.
    """
    results: List[List[float]] = []
    for t in texts:
        results.append(embed_text(t, host=host, model=model))
    return results

