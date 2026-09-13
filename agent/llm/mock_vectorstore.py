"""
agent/llm/mock_vectorstore.py

PERSON B: use this until Milestone 2 (real integration with Person A's
indexing/vectorstore/store.py). Matches the VectorStore interface exactly
so swapping the real one in later is a one-line change, not a rewrite.
"""

from typing import List, Optional, Dict
from shared.interfaces import VectorStore
from shared.schemas import Candidate


class MockVectorStore(VectorStore):
    def query(
        self,
        embedding: List[float],
        top_k: int = 8,
        filters: Optional[Dict[str, str]] = None,
    ) -> List[Candidate]:
        # TODO(person B): return canned fixtures during development,
        # e.g. simulate a known duplicate hit for testing Node 3/4.
        return [
            Candidate(
                chunk_id="mock_chunk_1",
                file_path="src/shared/formatDateUtils.ts",
                symbol_name="parse",
                code_snippet="export function parse(raw: string): Date { /* ... */ }",
                source_type="code",
                similarity=0.86,
            )
        ]

    def get_index_status(self) -> Dict[str, str]:
        return {"last_indexed_sha": "mock", "chunk_count": "1"}
