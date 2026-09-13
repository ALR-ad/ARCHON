"""
shared/config.py -- co-owned. Loads .agent-review.yml + env vars.
Pin values here that BOTH tracks must agree on.
"""
EMBEDDING_DIM = 768          # must match the Ollama embedding model's output dim
EMBEDDING_MODEL = "nomic-embed-text"
EVAL_MODEL = "qwen2.5-coder:32b"
SIMILARITY_FLOOR = 0.78
CONFIDENCE_THRESHOLD = 0.75
