import tempfile
from pathlib import Path
import pytest

from indexing.ingest.wiki_source import WikiSource, extract_title


def test_wiki_source_root_and_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        (root / "ARCHITECTURE.md").write_text("# System Architecture\nDetails here.")
        (root / "CONTRIBUTING.md").write_text("# Contributing Guide\nRules here.")
        (root / "docs").mkdir()
        (root / "docs" / "conventions.md").write_text("# Coding Conventions\nRules.")
        (root / "random.txt").write_text("not doc")

        ws = WikiSource(repo_root=str(root))
        docs = ws.get_all_documents()
        doc_paths = [d.file_path for d in docs]

        assert "ARCHITECTURE.md" in doc_paths
        assert "CONTRIBUTING.md" in doc_paths
        assert "docs/conventions.md" in doc_paths
        assert not any("random.txt" in p for p in doc_paths)


def test_wiki_source_custom_path():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        custom_wiki = root / "custom_wiki"
        custom_wiki.mkdir()
        (custom_wiki / "rules.md").write_text("# Custom Rules\nRule 1.")

        ws = WikiSource(repo_root=str(root), wiki_path=str(custom_wiki), include_root_docs=False)
        docs = ws.get_all_documents()

        assert len(docs) == 1
        assert "rules.md" in docs[0].file_path
        assert docs[0].title == "Custom Rules"
        assert docs[0].source_type == "wiki"


def test_wiki_source_confluence_directory_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        conf_dir = root / "confluence_export"
        conf_dir.mkdir()
        (conf_dir / "page1.html").write_text("<html><head><title>Architecture Decisions</title></head><body><h1>ADR-1</h1><p>Use Postgres.</p></body></html>")
        (conf_dir / "guide.md").write_text("# Deployment Guide\nDeploy via Docker.")

        ws = WikiSource(repo_root=str(root), confluence_export_path=str(conf_dir), include_root_docs=False)
        docs = ws.get_all_documents()

        assert len(docs) == 2
        titles = {d.title for d in docs}
        assert "Architecture Decisions" in titles
        assert "Deployment Guide" in titles

        html_doc = next(d for d in docs if "page1.html" in d.file_path)
        assert "Use Postgres" in html_doc.content
        assert html_doc.source_type == "wiki"


def test_wiki_source_confluence_zip_export():
    import zipfile

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        zip_path = root / "confluence_export.zip"

        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr(
                "spaces/DEV/Architecture.html",
                "<html><head><title>Microservices Guidelines</title></head><body><h1>Services</h1><p>Always use gRPC.</p></body></html>",
            )
            z.writestr(
                "spaces/DEV/Onboarding.md",
                "# Onboarding Checklist\nClone the repository.",
            )

        ws = WikiSource(repo_root=str(root), confluence_export_path=str(zip_path), include_root_docs=False)
        docs = ws.get_all_documents()

        assert len(docs) == 2
        titles = {d.title for d in docs}
        assert "Microservices Guidelines" in titles
        assert "Onboarding Checklist" in titles

        arch_doc = next(d for d in docs if "Architecture.html" in d.file_path)
        assert "Always use gRPC" in arch_doc.content
        assert arch_doc.source_type == "wiki"

