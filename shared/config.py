"""
shared/config.py -- co-owned. Loads .agent-review.yml + env vars.
Pin values here that BOTH tracks must agree on.
"""
EMBEDDING_DIM = 768          # must match the Ollama embedding model's output dim
EMBEDDING_MODEL = "nomic-embed-text"
EVAL_MODEL = "qwen2.5-coder:7b"
SIMILARITY_FLOOR = 0.78
CONFIDENCE_THRESHOLD = 0.75
RENAME_SIMILARITY_THRESHOLD = 0.97

# Wiki rules (architectural check). Prose rules and code never score high
# against each other, so rules are not filtered by similarity: Node 2 takes the
# WIKI_TOP_K closest rules per changed unit and the LLM decides.
WIKI_TOP_K = 4
# nomic-embed-text task prefixes. Measured on sample rules: with them the
# relevant rule ranks in the top 2; without, as low as 4th. Code-to-code
# search stays unprefixed. Both tracks must use the same pair.
WIKI_QUERY_PREFIX = "search_query: "
WIKI_DOCUMENT_PREFIX = "search_document: "
# Bump when the way wiki chunks are embedded changes; the indexer then
# re-embeds all wiki chunks on its next run.
WIKI_EMBEDDING_VERSION = "prefixed-v1"
