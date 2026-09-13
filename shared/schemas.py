"""
shared/schemas.py

THE CONTRACT. Both Person A (indexing) and Person B (agent) import from
here and NOTHING ELSE defines these shapes. Any change to this file
needs both people's review on the PR.
"""

from typing import List, Literal, Optional
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Produced by: Person B (agent/nodes/node1_ingest_diff.py)
# Consumed by: Person B (Node 2) -- passed to embed_text()
# ---------------------------------------------------------------------------
class ChangedCodeUnit(BaseModel):
    unit_id: str                              # stable hash of file_path+symbol_name+pr_sha
    file_path: str                            # e.g. "src/orders/parseOrderDate.ts"
    symbol_name: str                          # e.g. "parseOrderDate"
    language: str                             # e.g. "typescript"
    start_line: int
    end_line: int
    code: str                                 # full source of the function/class
    diff_type: Literal["added", "modified"]


# ---------------------------------------------------------------------------
# Produced by: Person A (indexing/embeddings/embedder.py) via embed_text()
# Consumed by: Person B (Node 2) as the query vector
# ---------------------------------------------------------------------------
class EmbeddedUnit(BaseModel):
    unit_id: str
    embedding: List[float]                    # fixed dimension, pinned in shared/config.py


# ---------------------------------------------------------------------------
# Produced by: Person A (indexing/vectorstore/store.py) via VectorStore.query()
# Consumed by: Person B (Node 2 -> Node 3)
# ---------------------------------------------------------------------------
class Candidate(BaseModel):
    chunk_id: str
    file_path: str
    symbol_name: str                          # empty string for wiki chunks
    code_snippet: str                         # source code OR wiki rule text
    source_type: Literal["code", "wiki"]
    similarity: float                         # cosine similarity, 0-1


# ---------------------------------------------------------------------------
# Produced by: Person B (Node 3, agent/llm/ollama_eval.py)
# Consumed by: Person B (Node 4)
# ---------------------------------------------------------------------------
class Finding(BaseModel):
    finding_id: str
    unit: ChangedCodeUnit
    candidates: List[Candidate]
    is_duplicate: bool
    is_architectural_violation: bool
    confidence: float                         # 0-1
    reasoning: str                            # <= 2 sentences
    suggested_replacement: Optional[str] = None
