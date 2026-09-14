import tempfile
from pathlib import Path
import pytest

from unittest.mock import patch, MagicMock
from indexing.ingest.repo_source import RepoSource, is_binary_file, matches_any_pattern, pull_github_repo


def test_repo_source_file_discovery():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "src").mkdir()
        (root / "src" / "main.py").write_text("def main(): pass")
        (root / "src" / "utils.ts").write_text("export function help() {}")
        (root / "README.txt").write_text("Hello")

        rs = RepoSource(repo_root=str(root))
        files = rs.get_all_files()
        paths = [f.file_path for f in files]

        assert "src/main.py" in paths
        assert "src/utils.ts" in paths
        assert "README.txt" not in paths  # Not in extension map


def test_repo_source_ignores_default_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "node_modules" / "pkg").mkdir(parents=True)
        (root / "node_modules" / "pkg" / "index.js").write_text("console.log(1);")
        (root / ".git").mkdir()
        (root / ".git" / "config.py").write_text("secret = 1")
        (root / "dist").mkdir()
        (root / "dist" / "bundle.js").write_text("bundled = 1")
        (root / "src").mkdir()
        (root / "src" / "app.py").write_text("print('ok')")

        rs = RepoSource(repo_root=str(root))
        files = rs.get_all_files()
        paths = [f.file_path for f in files]

        assert "src/app.py" in paths
        assert not any("node_modules" in p for p in paths)
        assert not any(".git" in p for p in paths)
        assert not any("dist" in p for p in paths)


def test_repo_source_ignores_binaries_and_locks():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "package-lock.json").write_text("{}")
        (root / "yarn.lock").write_text("")
        (root / "app.py").write_text("print(1)")
        # Binary null byte file
        (root / "corrupt.py").write_bytes(b"\x00\x01\x02")

        rs = RepoSource(repo_root=str(root))
        files = rs.get_all_files()
        paths = [f.file_path for f in files]

        assert "app.py" in paths
        assert "package-lock.json" not in paths
        assert "yarn.lock" not in paths
        assert "corrupt.py" not in paths


def test_repo_source_custom_ignores():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "test_fixtures").mkdir()
        (root / "test_fixtures" / "fixture.py").write_text("x = 1")
        (root / "real.py").write_text("x = 2")

        rs = RepoSource(repo_root=str(root), ignore_patterns=["**/test_fixtures/**"])
        files = rs.get_all_files()
        paths = [f.file_path for f in files]

        assert "real.py" in paths
        assert "test_fixtures/fixture.py" not in paths


def test_pull_github_repo_clone():
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "target_repo"
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result_path = pull_github_repo(
                repo="org/project",
                dest_dir=str(dest),
                branch="main",
                token="ghp_test123",
            )
            assert result_path == dest.resolve()
            assert mock_run.call_count == 1
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "git"
            assert cmd[1] == "clone"
            assert "--branch" in cmd
            assert "main" in cmd
            assert "https://x-access-token:ghp_test123@github.com/org/project.git" in cmd


def test_pull_github_repo_existing():
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "target_repo"
        dest.mkdir()
        (dest / ".git").mkdir()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            result_path = pull_github_repo(
                repo="org/project",
                dest_dir=str(dest),
            )
            assert result_path == dest.resolve()
            assert mock_run.call_count == 1
            cmd = mock_run.call_args[0][0]
            assert cmd == ["git", "-C", str(dest.resolve()), "pull"]

