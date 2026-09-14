"""
indexing/incremental.py

PERSON A: tracks last_indexed_sha per repo, re-embeds only changed files.
Handles additions, modifications, deletions, and renames.
Cleans up stale vectors for removed/renamed files and avoids re-embedding
unchanged files.
"""

from dataclasses import dataclass, field
import hashlib
import json
import logging
import os
from pathlib import Path
import subprocess
from typing import Callable, Dict, List, Optional, Set, Tuple, Any

from indexing.chunking.code_chunker import chunk_code
from indexing.chunking.wiki_chunker import chunk_wiki
from indexing.embeddings.embedder import embed_text, init_cache, save_cache
from indexing.ingest.repo_source import RepoSource, SourceFile
from indexing.ingest.wiki_source import WikiDocument, WikiSource
from indexing.chunking.wiki_chunker import chunk_wiki
from indexing.embeddings.embedder import embed_text, init_cache, save_cache
from indexing.ingest.repo_source import RepoSource, SourceFile
from indexing.ingest.wiki_source import WikiDocument, WikiSource
from indexing.vectorstore.store import QdrantVectorStore
from indexing.config import QDRANT_PATH

logger = logging.getLogger(__name__)

MANIFEST_FILENAME = "indexing_manifest.json"


@dataclass
class IndexingStats:
    added_files: int = 0
    modified_files: int = 0
    deleted_files: int = 0
    renamed_files: int = 0
    unchanged_files: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    last_indexed_sha: str = ""


class IncrementalIndexer:
    """
    Coordinates incremental indexing of codebase and wiki documentation into Qdrant.
    """

    def __init__(
        self,
        repo_root: Optional[str] = ".",
        github_repo: Optional[str] = None,
        wiki_path: Optional[str] = None,
        confluence_export_path: Optional[str] = None,
        db_path: Optional[str] = None,
        vector_store: Optional[QdrantVectorStore] = None,
        embed_fn: Optional[Callable[[str], List[float]]] = None,
    ):
        self.github_repo = github_repo
        self.repo_root = Path(repo_root).resolve() if repo_root else None
        self.wiki_path = wiki_path
        self.confluence_export_path = confluence_export_path
        self.vector_store = vector_store or QdrantVectorStore(path=db_path)
        self.embed_fn = embed_fn or embed_text

        base_dir = self.vector_store.db_path or Path(QDRANT_PATH).resolve()
        base_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = base_dir / MANIFEST_FILENAME

        # Initialize embedding cache in the db directory
        init_cache(str(base_dir))

    def close(self) -> None:
        """Close underlying vector store."""
        if hasattr(self, "vector_store") and self.vector_store is not None:
            self.vector_store.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


    def _load_manifest(self) -> Dict[str, Any]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load manifest: {e}")
        return {"files": {}, "last_sha": ""}

    def _save_manifest(self, manifest: Dict[str, Any]) -> None:
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save manifest: {e}")

    def get_git_head_sha(self) -> Optional[str]:
        """Get current git commit SHA if repo_root is a git repository."""
        try:
            res = subprocess.run(
                ["git", "-C", str(self.repo_root), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            )
            sha = res.stdout.strip()
            return sha if len(sha) == 40 else None
        except Exception:
            return None

    def get_git_diff_status(self, base_sha: str, head_sha: str) -> Optional[Dict[str, str]]:
        """
        Use git diff to identify added, modified, deleted, and renamed files.
        Returns dict mapping file_path -> status ('A', 'M', 'D', 'R:old_path').
        """
        try:
            res = subprocess.run(
                ["git", "-C", str(self.repo_root), "diff", "--name-status", "-M", base_sha, head_sha],
                capture_output=True,
                text=True,
                check=True,
            )
            diff_map: Dict[str, str] = {}
            for line in res.stdout.splitlines():
                parts = line.strip().split("\t")
                if not parts:
                    continue
                status_code = parts[0]
                if status_code.startswith("R"):
                    # Rename: R100\told_path\tnew_path
                    old_path = parts[1]
                    new_path = parts[2]
                    diff_map[new_path] = f"R:{old_path}"
                elif status_code == "A":
                    diff_map[parts[1]] = "A"
                elif status_code == "M":
                    diff_map[parts[1]] = "M"
                elif status_code == "D":
                    diff_map[parts[1]] = "D"
            return diff_map
        except Exception as e:
            logger.debug(f"Git diff failed between {base_sha} and {head_sha}: {e}")
            return None

    def run_indexing(self, force_full: bool = False) -> IndexingStats:
        """
        Execute full or incremental indexing run.
        """
        stats = IndexingStats()
        manifest = self._load_manifest()
        prev_files: Dict[str, str] = {} if force_full else manifest.get("files", {})
        last_sha = "" if force_full else manifest.get("last_sha", "")
        current_sha = self.get_git_head_sha() or ""

        # Step 1: Discover all current files (code + docs)
        repo_source = RepoSource(
            repo_root=str(self.repo_root) if self.repo_root else None,
            github_repo=self.github_repo,
        )
        current_source_files: Dict[str, SourceFile] = {
            sf.file_path: sf for sf in repo_source.discover_files()
        }
        self.repo_root = repo_source.repo_root
        effective_root = str(repo_source.repo_root)

        wiki_source = WikiSource(
            repo_root=effective_root,
            wiki_path=self.wiki_path,
            confluence_export_path=self.confluence_export_path,
        )
        current_wiki_docs: Dict[str, WikiDocument] = {
            wd.file_path: wd for wd in wiki_source.discover_documents()
        }

        # Combine all indexable file paths and their content hashes
        all_current_paths: Set[str] = set(current_source_files.keys()) | set(current_wiki_docs.keys())
        current_file_hashes: Dict[str, str] = {}
        for p, sf in current_source_files.items():
            current_file_hashes[p] = sf.content_hash
        for p, wd in current_wiki_docs.items():
            current_file_hashes[p] = wd.content_hash

        # Step 2: Determine file diffs (Added, Modified, Deleted, Renamed, Unchanged)
        git_diff_map = None
        if not force_full and last_sha and current_sha and last_sha != current_sha:
            git_diff_map = self.get_git_diff_status(last_sha, current_sha)

        added: Set[str] = set()
        modified: Set[str] = set()
        deleted: Set[str] = set()
        renamed: Dict[str, str] = {}  # new_path -> old_path
        unchanged: Set[str] = set()

        if force_full or not prev_files:
            # Full initial index
            added = set(all_current_paths)
        elif git_diff_map is not None:
            # Git-accelerated diffing
            for path, status in git_diff_map.items():
                if status == "A" and path in all_current_paths:
                    added.add(path)
                elif status == "M" and path in all_current_paths:
                    modified.add(path)
                elif status == "D":
                    deleted.add(path)
                elif status.startswith("R:"):
                    old_path = status.split(":", 1)[1]
                    renamed[path] = old_path

            # Also check any file in prev_files that doesn't exist anymore
            for p in prev_files:
                if p not in all_current_paths and p not in deleted and p not in renamed.values():
                    deleted.add(p)

            # Any current file not in added/modified/renamed is unchanged
            for p in all_current_paths:
                if p not in added and p not in modified and p not in renamed:
                    unchanged.add(p)
        else:
            # Hash-based diffing (reliable across git and non-git environments)
            # Check for deletions
            for prev_p, prev_h in prev_files.items():
                if prev_p not in all_current_paths:
                    # Check if it was renamed (same content hash in a new file)
                    matching_new = [
                        new_p for new_p, new_h in current_file_hashes.items()
                        if new_h == prev_h and new_p not in prev_files
                    ]
                    if matching_new:
                        new_p = matching_new[0]
                        renamed[new_p] = prev_p
                    else:
                        deleted.add(prev_p)

            # Check current files against prev_files
            for curr_p, curr_h in current_file_hashes.items():
                if curr_p in renamed:
                    continue
                if curr_p not in prev_files:
                    added.add(curr_p)
                elif prev_files[curr_p] != curr_h:
                    modified.add(curr_p)
                else:
                    unchanged.add(curr_p)

        stats.added_files = len(added)
        stats.modified_files = len(modified)
        stats.deleted_files = len(deleted)
        stats.renamed_files = len(renamed)
        stats.unchanged_files = len(unchanged)

        logger.info(
            f"Indexing summary: {len(added)} added, {len(modified)} modified, "
            f"{len(deleted)} deleted, {len(renamed)} renamed, {len(unchanged)} unchanged."
        )

        # Step 3: Handle Deletions and Renames in Vector Store
        files_to_remove = set(deleted)
        for new_p, old_p in renamed.items():
            files_to_remove.add(old_p)
        for mod_p in modified:
            files_to_remove.add(mod_p)

        if files_to_remove:
            self.vector_store.delete_by_file_paths(list(files_to_remove))
            stats.chunks_deleted = len(files_to_remove)

        # Step 4: Chunk, Embed, and Upsert Files (added + modified + renamed new paths)
        files_to_process = added | modified | set(renamed.keys())
        chunks_to_upsert: List[Dict[str, Any]] = []

        for path in files_to_process:
            if path in current_source_files:
                sf = current_source_files[path]
                units = chunk_code(
                    file_path=sf.file_path,
                    code=sf.content,
                    language=sf.language,
                    commit_sha=current_sha,
                    diff_type="modified" if path in modified else "added",
                )
                for u in units:
                    vec = self.embed_fn(u.code)
                    chunks_to_upsert.append({
                        "chunk_id": u.unit_id,
                        "file_path": u.file_path,
                        "symbol_name": u.symbol_name,
                        "code_snippet": u.code,
                        "source_type": "code",
                        "language": u.language,
                        "start_line": u.start_line,
                        "end_line": u.end_line,
                        "commit_sha": current_sha,
                        "content_hash": hashlib.sha256(u.code.encode("utf-8")).hexdigest(),
                        "vector": vec,
                    })
            elif path in current_wiki_docs:
                wd = current_wiki_docs[path]
                w_chunks = chunk_wiki(content=wd.content, file_path=wd.file_path)
                for wc in w_chunks:
                    vec = self.embed_fn(wc.code_snippet)
                    chunks_to_upsert.append({
                        "chunk_id": wc.chunk_id,
                        "file_path": wc.file_path,
                        "symbol_name": wc.symbol_name,
                        "code_snippet": wc.code_snippet,
                        "source_type": "wiki",
                        "language": "markdown",
                        "start_line": wc.start_line,
                        "end_line": wc.end_line,
                        "commit_sha": current_sha,
                        "content_hash": wc.content_hash,
                        "vector": vec,
                    })

        if chunks_to_upsert:
            self.vector_store.upsert(chunks_to_upsert)
            stats.chunks_upserted = len(chunks_to_upsert)

        # Step 5: Save manifest and status
        new_manifest = {
            "files": current_file_hashes,
            "last_sha": current_sha,
        }
        self._save_manifest(new_manifest)
        if current_sha:
            self.vector_store.set_last_indexed_sha(current_sha)
        stats.last_indexed_sha = current_sha

        # Save persistent embedding cache
        save_cache()

        return stats

