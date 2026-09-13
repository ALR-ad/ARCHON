"""
indexing/ingest/repo_source.py

PERSON A: pulls files from a GitHub repo (via local clone/directory).
Enumerates relevant source files, filters ignored paths and binaries,
and preserves file paths and metadata for downstream chunking.
"""

from dataclasses import dataclass
import fnmatch
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from typing import Dict, Iterator, List, Optional, Set

from indexing.config import load_agent_review_config

logger = logging.getLogger(__name__)

# Map extensions to tree-sitter language names
EXTENSION_LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".cs": "c_sharp",
    ".rb": "ruby",
    ".php": "php",
}

DEFAULT_IGNORE_DIRS: Set[str] = {
    ".git",
    "node_modules",
    "vendor",
    "build",
    "dist",
    "__pycache__",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    "data",
}

DEFAULT_IGNORE_PATTERNS: List[str] = [
    "**/node_modules/**",
    "**/*.lock",
    "**/package-lock.json",
    "**/yarn.lock",
    "**/pnpm-lock.yaml",
    "**/dist/**",
    "**/build/**",
    "**/.git/**",
    "**/vendor/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/*.pyo",
    "**/*.min.js",
    "**/*.min.css",
]

BINARY_EXTENSIONS: Set[str] = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".bz2",
    ".exe", ".dll", ".so", ".dylib", ".bin",
    ".pyc", ".pyo", ".pyd", ".whl",
    ".ttf", ".woff", ".woff2", ".eot",
    ".mp3", ".mp4", ".wav", ".avi", ".mov",
    ".parquet", ".lance",
}


@dataclass
class SourceFile:
    file_path: str               # Relative path, e.g. "src/orders/parseOrderDate.ts"
    abs_path: str                # Absolute filesystem path
    language: str                # e.g. "typescript", "python"
    content: str                 # Full file text
    content_hash: str            # SHA-256 hash of content
    size_bytes: int


def is_binary_file(path: Path) -> bool:
    """Check whether a file is binary by extension or null-byte heuristic."""
    if path.suffix.lower() in BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            if b"\x00" in chunk:
                return True
    except Exception:
        return True
    return False


def normalize_rel_path(path: Path, root: Path) -> str:
    """Normalize relative path using forward slashes."""
    rel = path.relative_to(root)
    return str(rel).replace("\\", "/")


def matches_any_pattern(path_str: str, patterns: List[str]) -> bool:
    """Check if normalized relative path matches any glob pattern."""
    normalized = path_str.replace("\\", "/")
    posix_path = "/" + normalized if not normalized.startswith("/") else normalized
    for pat in patterns:
        clean_pat = pat.replace("\\", "/")
        if fnmatch.fnmatch(normalized, clean_pat):
            return True
        if fnmatch.fnmatch(posix_path, clean_pat):
            return True
        if fnmatch.fnmatch(os.path.basename(normalized), clean_pat):
            return True
    return False


def pull_github_repo(
    repo: str,
    dest_dir: Optional[str] = None,
    branch: Optional[str] = None,
    token: Optional[str] = None,
) -> Path:
    """
    Pulls / clones a GitHub repository locally for indexing.
    Input:
        repo: "owner/repo" or "https://github.com/owner/repo.git"
    """
    token = token or os.getenv("GITHUB_TOKEN")
    if repo.startswith("https://") or repo.startswith("git@"):
        clone_url = repo
        repo_name = repo.rstrip("/").split("/")[-1].replace(".git", "")
    else:
        repo_name = repo.split("/")[-1]
        if token:
            clone_url = f"https://x-access-token:{token}@github.com/{repo}.git"
        else:
            clone_url = f"https://github.com/{repo}.git"

    target_dir = Path(dest_dir or f"./data/repos/{repo_name}").resolve()
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if target_dir.exists() and (target_dir / ".git").exists():
        logger.info(f"Pulling latest changes in existing repo at {target_dir}")
        subprocess.run(["git", "-C", str(target_dir), "pull"], check=True, capture_output=True)
    else:
        logger.info(f"Cloning {repo} to {target_dir}")
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd.extend(["--branch", branch])
        cmd.extend([clone_url, str(target_dir)])
        subprocess.run(cmd, check=True, capture_output=True)

    return target_dir


class RepoSource:
    """
    Scans a repository (local or pulled from GitHub) for indexable source code files.
    """

    def __init__(
        self,
        repo_root: Optional[str] = None,
        github_repo: Optional[str] = None,
        ignore_patterns: Optional[List[str]] = None,
        custom_extensions: Optional[Dict[str, str]] = None,
    ):
        if github_repo:
            self.repo_root = pull_github_repo(github_repo)
        else:
            self.repo_root = Path(repo_root or ".").resolve()
        cfg = load_agent_review_config(str(self.repo_root))
        cfg_ignores = cfg.get("ignore_paths", [])

        self.ignore_patterns: List[str] = list(DEFAULT_IGNORE_PATTERNS)
        if cfg_ignores:
            self.ignore_patterns.extend(cfg_ignores)
        if ignore_patterns:
            self.ignore_patterns.extend(ignore_patterns)

        self.ext_map = dict(EXTENSION_LANGUAGE_MAP)
        if custom_extensions:
            self.ext_map.update(custom_extensions)

    def is_ignored(self, path: Path) -> bool:
        """Check if file/directory path is ignored."""
        try:
            rel = path.relative_to(self.repo_root)
        except ValueError:
            return True

        parts = rel.parts
        for part in parts:
            if part in DEFAULT_IGNORE_DIRS:
                return True

        rel_str = normalize_rel_path(path, self.repo_root)
        return matches_any_pattern(rel_str, self.ignore_patterns)

    def discover_files(self) -> Iterator[SourceFile]:
        """
        Recursively discover all indexable source files in the repository.
        """
        if not self.repo_root.exists():
            return

        for root_dir, dirs, files in os.walk(self.repo_root):
            # Prune ignored directories in-place for efficiency
            dirs[:] = [
                d for d in dirs
                if d not in DEFAULT_IGNORE_DIRS
                and not matches_any_pattern(
                    normalize_rel_path(Path(root_dir) / d, self.repo_root),
                    self.ignore_patterns,
                )
            ]

            for file_name in files:
                file_path = Path(root_dir) / file_name
                ext = file_path.suffix.lower()

                if ext not in self.ext_map:
                    continue

                if self.is_ignored(file_path):
                    continue

                if is_binary_file(file_path):
                    continue

                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()

                    rel_path = normalize_rel_path(file_path, self.repo_root)
                    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    yield SourceFile(
                        file_path=rel_path,
                        abs_path=str(file_path),
                        language=self.ext_map[ext],
                        content=content,
                        content_hash=content_hash,
                        size_bytes=len(content.encode("utf-8")),
                    )
                except Exception:
                    continue

    def get_all_files(self) -> List[SourceFile]:
        """Convenience method returning list of all discovered files."""
        return list(self.discover_files())

