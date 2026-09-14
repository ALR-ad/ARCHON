"""
indexing/config.py

Person A / Track A configuration loader.
Reads .env and .agent-review.yml while honoring shared/config.py constants.
"""

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from dotenv import load_dotenv

from shared.config import (
    CONFIDENCE_THRESHOLD,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    EVAL_MODEL,
    SIMILARITY_FLOOR,
)

# Load .env if present
load_dotenv()

# Track A environment settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", None)  # e.g. "http://localhost:6333" for Docker
QDRANT_PATH = os.getenv("QDRANT_PATH", os.getenv("VECTOR_STORE_PATH", "./data/qdrant"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)


def load_agent_review_config(repo_root: str = ".") -> Dict[str, Any]:
    """
    Loads .agent-review.yml from repo_root if it exists, otherwise falls back
    to .agent-review.yml.example or sensible defaults.
    """
    root = Path(repo_root)
    cfg_file = root / ".agent-review.yml"
    if not cfg_file.exists():
        cfg_file = root / ".agent-review.yml.example"

    if cfg_file.exists():
        try:
            with open(cfg_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception:
            pass

    return {
        "similarity_floor": SIMILARITY_FLOOR,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "ignore_paths": [
            "**/node_modules/**",
            "**/*.lock",
            "**/dist/**",
            "**/.git/**",
            "**/vendor/**",
            "**/build/**",
            "**/__pycache__/**",
        ],
        "enabled_checks": {
            "duplication": True,
            "architectural_violation": True,
        },
    }
