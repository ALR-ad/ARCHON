"""
tests/integration/test_similarity_tuning.py

A calibration test suite for the SIMILARITY_FLOOR (currently 0.78).
This uses the real Ollama embedding model (nomic-embed-text) to evaluate
cosine similarity between controlled pairs of code snippets.

It acts as a regression test to ensure that the chosen floor correctly
separates true positives (duplicates/refactors) from true negatives
(novel logic, unrelated logic).
"""

import math
import pytest
from typing import List

from shared.config import SIMILARITY_FLOOR

# Only run if integration tests are enabled and models are available
pytestmark = pytest.mark.integration


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


@pytest.fixture(scope="module")
def embed_fn():
    """Import the real embed_text function."""
    from indexing.embeddings.embedder import embed_text
    from tests.integration.test_end_to_end_real import _ollama_reachable, _model_available
    
    if not _ollama_reachable() or not _model_available("nomic-embed-text"):
        pytest.skip("Ollama or nomic-embed-text not available")
        
    return embed_text


class TestSimilarityCalibration:
    
    def test_exact_duplicate_is_high(self, embed_fn):
        """Exact matches should be nearly 1.0."""
        code = "def add(a, b):\n    return a + b"
        sim = cosine_similarity(embed_fn(code), embed_fn(code))
        assert sim > 0.99

    def test_renamed_duplicate_above_floor(self, embed_fn):
        """A function where only the name and variables changed should be above floor."""
        code1 = "def parse_date(raw_str):\n    parts = raw_str.split('-')\n    return datetime(int(parts[0]), int(parts[1]), int(parts[2]))"
        code2 = "def get_order_date(date_string):\n    p = date_string.split('-')\n    return datetime(int(p[0]), int(p[1]), int(p[2]))"
        
        sim = cosine_similarity(embed_fn(code1), embed_fn(code2))
        # Expect very high similarity
        assert sim >= SIMILARITY_FLOOR
        
    def test_different_logic_below_floor(self, embed_fn):
        """Completely different logic should fall below the floor."""
        code1 = "def parse_date(raw_str):\n    parts = raw_str.split('-')\n    return datetime(int(parts[0]), int(parts[1]), int(parts[2]))"
        code2 = "def calculate_tax(amount, rate):\n    return amount * rate"
        
        sim = cosine_similarity(embed_fn(code1), embed_fn(code2))
        assert sim < SIMILARITY_FLOOR
        
    def test_same_keywords_different_logic_below_floor(self, embed_fn):
        """Code that uses similar domain words but does different things should be below floor if possible."""
        code1 = "def get_order_date(order):\n    return order.created_at"
        code2 = "def validate_order(order):\n    if not order.items:\n        raise ValueError('Empty order')"
        
        sim = cosine_similarity(embed_fn(code1), embed_fn(code2))
        assert sim < SIMILARITY_FLOOR

    def test_translated_language_above_or_near_floor(self, embed_fn):
        """A Javascript vs Python version of the same logic."""
        python_code = "def add(a, b):\n    return a + b"
        js_code = "function add(a, b) {\n    return a + b;\n}"
        
        sim = cosine_similarity(embed_fn(python_code), embed_fn(js_code))
        # nomic-embed-text is decent at cross-language. 
        # Even if it's slightly below floor, it should be relatively high.
        # We assert it's at least > 0.70 to ensure the model behaves reasonably.
        assert sim > 0.70
