import os
import tempfile
from pathlib import Path
import pytest

from indexing.embeddings.embedder import deterministic_mock_embed
from indexing.incremental import IncrementalIndexer
from indexing.vectorstore.store import QdrantVectorStore


def test_incremental_indexing_workflow():
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_dir = Path(tmpdir) / "repo"
        db_dir = Path(tmpdir) / "db"
        repo_dir.mkdir()
        db_dir.mkdir()

        # Step 1: Create initial files
        file1 = repo_dir / "math_utils.py"
        file1.write_text("def add(a: int, b: int) -> int:\n    return a + b\n")

        file2 = repo_dir / "string_utils.py"
        file2.write_text("def capitalize(s: str) -> str:\n    return s.capitalize()\n")

        doc1 = repo_dir / "ARCHITECTURE.md"
        doc1.write_text("# Architecture\n- Rule: Use math_utils for arithmetic.\n")

        store = QdrantVectorStore(path=str(db_dir))
        indexer = IncrementalIndexer(
            repo_root=str(repo_dir),
            vector_store=store,
            embed_fn=deterministic_mock_embed,
        )

        # Initial Run (Full Index)
        s1 = indexer.run_indexing()
        assert s1.added_files == 3
        assert s1.unchanged_files == 0
        assert s1.chunks_upserted >= 3
        assert store.count() >= 3

        # Step 2: Second Run with No Changes
        s2 = indexer.run_indexing()
        assert s2.added_files == 0
        assert s2.modified_files == 0
        assert s2.unchanged_files == 3
        assert s2.chunks_upserted == 0

        # Step 3: Modify one file
        file1.write_text("def add(a: int, b: int) -> int:\n    \"\"\"Updated doc\"\"\"\n    return a + b\n")
        s3 = indexer.run_indexing()
        assert s3.modified_files == 1
        assert s3.unchanged_files == 2
        assert s3.chunks_upserted >= 1

        # Step 4: Delete a file
        file2.unlink()
        s4 = indexer.run_indexing()
        assert s4.deleted_files == 1
        assert s4.unchanged_files == 2
        # Verify deleted file chunks are purged from vector store
        remaining_candidates = store.query(deterministic_mock_embed("capitalize"), top_k=10)
        assert not any("string_utils.py" in c.file_path for c in remaining_candidates)

        # Step 5: Rename a file (simulated move: same content in new file)
        content = file1.read_text()
        file1.unlink()
        new_file = repo_dir / "arithmetic.py"
        new_file.write_text(content)

        s5 = indexer.run_indexing()
        assert s5.renamed_files == 1
        assert s5.deleted_files == 0
        # Verify old file chunks removed and new file present
        all_candidates = store.query(deterministic_mock_embed("add"), top_k=10)
        assert not any("math_utils.py" in c.file_path for c in all_candidates)
        assert any("arithmetic.py" in c.file_path for c in all_candidates)
        indexer.close()

