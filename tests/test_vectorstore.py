import tempfile
import pytest
from pathlib import Path

from indexing.embeddings.embedder import deterministic_mock_embed
from indexing.vectorstore.store import QdrantVectorStore
from shared.config import EMBEDDING_DIM
from shared.schemas import Candidate


def test_vectorstore_insert_and_query():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = QdrantVectorStore(path=tmpdir)
        vec1 = deterministic_mock_embed("parseOrderDate function")
        vec2 = deterministic_mock_embed("formatDate helper")

        records = [
            {
                "chunk_id": "c1",
                "file_path": "src/orders/parseOrderDate.ts",
                "symbol_name": "parseOrderDate",
                "code_snippet": "export function parseOrderDate(raw: string) {}",
                "source_type": "code",
                "language": "typescript",
                "start_line": 1,
                "end_line": 3,
                "vector": vec1,
            },
            {
                "chunk_id": "c2",
                "file_path": "src/utils/formatDate.ts",
                "symbol_name": "formatDate",
                "code_snippet": "export function formatDate(d: Date) {}",
                "source_type": "code",
                "language": "typescript",
                "start_line": 1,
                "end_line": 3,
                "vector": vec2,
            },
        ]

        inserted = store.upsert(records)
        assert inserted == 2
        assert store.count() == 2

        # Query with vec1
        results = store.query(embedding=vec1, top_k=2)
        assert len(results) == 2
        assert isinstance(results[0], Candidate)
        assert results[0].chunk_id == "c1"
        assert results[0].symbol_name == "parseOrderDate"
        assert results[0].similarity >= 0.99  # Identical vector ~ 1.0
        store.close()



def test_vectorstore_metadata_filtering():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = QdrantVectorStore(path=tmpdir)
        v = deterministic_mock_embed("test content")

        records = [
            {
                "chunk_id": "code_1",
                "file_path": "src/service.py",
                "symbol_name": "serve",
                "code_snippet": "def serve(): pass",
                "source_type": "code",
                "vector": v,
            },
            {
                "chunk_id": "wiki_1",
                "file_path": "docs/ARCHITECTURE.md",
                "symbol_name": "",
                "code_snippet": "[Architecture] Controllers must not call DB",
                "source_type": "wiki",
                "vector": v,
            },
        ]
        store.upsert(records)

        code_matches = store.query(embedding=v, filters={"source_type": "code"})
        assert len(code_matches) == 1
        assert code_matches[0].source_type == "code"
        assert code_matches[0].chunk_id == "code_1"

        wiki_matches = store.query(embedding=v, filters={"source_type": "wiki"})
        assert len(wiki_matches) == 1
        assert wiki_matches[0].source_type == "wiki"
        assert wiki_matches[0].symbol_name == ""
        store.close()


def test_vectorstore_delete_and_status():
    with tempfile.TemporaryDirectory() as tmpdir:
        store = QdrantVectorStore(path=tmpdir)
        v = deterministic_mock_embed("dummy")

        records = [
            {"chunk_id": "c1", "file_path": "src/a.py", "code_snippet": "a", "source_type": "code", "vector": v},
            {"chunk_id": "c2", "file_path": "src/b.py", "code_snippet": "b", "source_type": "code", "vector": v},
        ]
        store.upsert(records)
        store.set_last_indexed_sha("abcdef1234567890abcdef1234567890abcdef12")

        status = store.get_index_status()
        assert status["chunk_count"] == "2"
        assert status["last_indexed_sha"] == "abcdef1234567890abcdef1234567890abcdef12"

        # Delete by file path
        store.delete_by_file_paths(["src/a.py"])
        assert store.count() == 1
        rem = store.query(v)
        assert len(rem) == 1
        assert rem[0].chunk_id == "c2"

        # Delete by chunk id
        store.delete_by_chunk_ids(["c2"])
        assert store.count() == 0
        store.close()

