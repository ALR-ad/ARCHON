"""
shared/interfaces.py

The ONLY two functions that cross the Person A / Person B boundary.
Person A implements these for real in indexing/. Person B imports this
module's type signatures and, until Milestone 2, uses a mock
implementation (see agent/llm/mock_vectorstore.py) matching the exact
same signature so the DAG can be built and tested independently.

Do not add new cross-boundary functions without updating this file
and getting both people's sign-off -- this is the whole point of the split.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from shared.schemas import Candidate


def embed_text(text: str) -> List[float]:
    """
    Owned by: Person A (indexing/embeddings/embedder.py)
    Called by: Person B (agent/nodes/node2_semantic_search.py)

    Wraps the local Ollama embedding model. MUST use the exact same
    model/version as whatever built the vector store index, or
    similarity scores become meaningless.

    Input:
        text: str -- raw code (a ChangedCodeUnit.code) or raw query text

    Output:
        List[float] -- embedding vector, fixed dimension (see
        shared/config.py: EMBEDDING_DIM). Same dimension every call.
    """
    from indexing.embeddings.embedder import embed_text as _impl
    return _impl(text)


class VectorStore(ABC):
    """
    Owned by: Person A (indexing/vectorstore/store.py)
    Called by: Person B (agent/nodes/node2_semantic_search.py)
    """

    @abstractmethod
    def query(
        self,
        embedding: List[float],
        top_k: int = 8,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Candidate]:
        """
        Input:
            embedding: List[float] -- output of embed_text(), same dimension
                        as the index
            top_k: int -- max results to return, default 8
            filters: Optional[Dict[str, str]] -- e.g. {"source_type": "code"}
                      or {"source_type": "wiki"}; None means no filter

        Output:
            List[Candidate] -- sorted descending by similarity.
            Empty list is a valid result (no findings for this unit).
        """
        raise NotImplementedError

    @abstractmethod
    def get_index_status(self) -> Dict[str, str]:
        """
        Optional health-check function, useful for debugging.

        Output: Dict with at least:
            {"last_indexed_sha": str, "chunk_count": str}
        """
        raise NotImplementedError
