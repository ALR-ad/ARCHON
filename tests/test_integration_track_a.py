import os
import tempfile
from pathlib import Path
import pytest

from indexing.embeddings.embedder import set_custom_embedder, deterministic_mock_embed
from indexing.incremental import IncrementalIndexer
from indexing.vectorstore.store import QdrantVectorStore
import shared.interfaces as shared_interfaces
from shared.schemas import Candidate


def test_end_to_end_track_a_indexing_and_retrieval():
    """
    Validates Track A end-to-end:
    - Codebase + wiki indexing
    - Shared VectorStore contract compliance
    - Person B Candidate retrieval requirements
    """
    set_custom_embedder(deterministic_mock_embed)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_dir = Path(tmpdir) / "archon_sample_repo"
            db_dir = Path(tmpdir) / "vector_db"
            repo_dir.mkdir()
            db_dir.mkdir()

            # Create sample repository code
            src_orders = repo_dir / "src" / "orders"
            src_orders.mkdir(parents=True)
            (src_orders / "parseOrderDate.ts").write_text("""
export function parseOrderDate(raw: string): Date {
    // legacy format parser
    return new Date(raw);
}
""")
            src_db = repo_dir / "src" / "db"
            src_db.mkdir(parents=True)
            (src_db / "client.py").write_text("""
class DatabaseClient:
    def execute_query(self, sql: str):
        return []
""")

            # Create architectural wiki document
            docs_dir = repo_dir / "docs"
            docs_dir.mkdir(parents=True)
            (docs_dir / "ARCHITECTURE.md").write_text("""# Architectural Standards

## Database Layer
- Rule: Controllers must not call the database layer directly.
- Rule: All database operations must go through service repositories.
""")

            # Run Track A Indexing
            vs = QdrantVectorStore(path=str(db_dir))
            indexer = IncrementalIndexer(
                repo_root=str(repo_dir),
                vector_store=vs,
                embed_fn=deterministic_mock_embed,
            )
            stats = indexer.run_indexing()

            assert stats.added_files >= 3
            assert vs.count() >= 3

            # Test Person B VectorStore Query Contract
            # 1. Query for date parsing utility (Code search)
            query_vec_code = shared_interfaces.embed_text("export function parseOrderDate(raw: string)")
            code_results = vs.query(
                embedding=query_vec_code,
                top_k=5,
                filters={"source_type": "code"},
            )
            assert len(code_results) >= 1
            top_code = code_results[0]
            assert isinstance(top_code, Candidate)
            assert top_code.source_type == "code"
            assert "parseOrderDate.ts" in top_code.file_path
            assert top_code.symbol_name == "parseOrderDate"
            assert top_code.similarity >= 0.70

            # 2. Query for architectural rule (Wiki search)
            query_vec_wiki = shared_interfaces.embed_text("Controllers must not call the database layer directly")
            wiki_results = vs.query(
                embedding=query_vec_wiki,
                top_k=5,
                filters={"source_type": "wiki"},
            )
            assert len(wiki_results) >= 1
            top_wiki = wiki_results[0]
            assert isinstance(top_wiki, Candidate)
            assert top_wiki.source_type == "wiki"
            assert top_wiki.symbol_name == ""
            assert "Controllers must not call the database layer directly" in top_wiki.code_snippet
            assert top_wiki.similarity >= 0.70

            # 3. Verify status output
            status = vs.get_index_status()
            assert "chunk_count" in status
            assert int(status["chunk_count"]) >= 3
            vs.close()
    finally:
        set_custom_embedder(None)
