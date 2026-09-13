"""
indexing/ingest/wiki_source.py

PERSON A: pulls wiki/style-guide/markdown pages for convention chunking.
Supports in-repo documentation files (ARCHITECTURE.md, CONTRIBUTING.md,
docs/, wiki/) as well as custom wiki directories.
"""

from dataclasses import dataclass
import hashlib
import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

logger = logging.getLogger(__name__)

WELL_KNOWN_DOC_FILES = [
    "ARCHITECTURE.md",
    "CONTRIBUTING.md",
    "STYLEGUIDE.md",
    "STYLE_GUIDE.md",
    "CODING_STANDARDS.md",
    "CONVENTIONS.md",
    "DESIGN.md",
]

WELL_KNOWN_DOC_DIRS = [
    "docs",
    "wiki",
    "documentation",
    ".github/wiki",
]


@dataclass
class WikiDocument:
    file_path: str       # Normalized relative path, e.g. "docs/ARCHITECTURE.md"
    abs_path: str        # Absolute path on disk
    title: str           # Document title (from filename or top heading)
    content: str         # Full markdown content
    content_hash: str    # SHA-256 hash of content
    source_type: str = "wiki"


def normalize_rel_path(path: Path, root: Path) -> str:
    """Normalize relative path using forward slashes."""
    try:
        rel = path.relative_to(root)
        return str(rel).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def extract_title(content: str, fallback: str = "") -> str:
    """Extract top-level markdown heading title or return fallback."""
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


from html.parser import HTMLParser
import zipfile


class HTMLToMarkdownParser(HTMLParser):
    """Simple parser that extracts text from HTML and converts headings/lists to markdown."""

    def __init__(self):
        super().__init__()
        self.output: List[str] = []
        self.title = ""
        self._in_title = False
        self._skip = False
        self._current_tag = ""

    def handle_starttag(self, tag, attrs):
        self._current_tag = tag
        if tag in ("script", "style", "nav", "footer"):
            self._skip = True
        elif tag == "title":
            self._in_title = True
        elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(tag[1])
            self.output.append(f"\n\n{'#' * level} ")
        elif tag == "p":
            self.output.append("\n\n")
        elif tag == "li":
            self.output.append("\n- ")
        elif tag == "blockquote":
            self.output.append("\n> ")

    def handle_endtag(self, tag):
        if tag in ("script", "style", "nav", "footer"):
            self._skip = False
        elif tag == "title":
            self._in_title = False
        elif tag in ("p", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.output.append("\n")

    def handle_data(self, data):
        if self._skip:
            return
        if self._in_title and not self.title:
            self.title = data.strip()
        text = data.strip()
        if text:
            self.output.append(data)

    def get_markdown(self) -> str:
        return "".join(self.output).strip()


def convert_html_to_markdown(html_content: str) -> Tuple[str, str]:
    """Returns (title, markdown_content)."""
    parser = HTMLToMarkdownParser()
    parser.feed(html_content)
    return parser.title, parser.get_markdown()


class WikiSource:
    """
    Discovers and reads indexable wiki, style-guide, architectural documents,
    and Confluence exports (HTML/Markdown/ZIP).
    """

    def __init__(
        self,
        repo_root: str = ".",
        wiki_path: Optional[str] = None,
        confluence_export_path: Optional[str] = None,
        include_root_docs: bool = True,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.wiki_path = Path(wiki_path).resolve() if wiki_path else None
        self.confluence_export_path = Path(confluence_export_path).resolve() if confluence_export_path else None
        self.include_root_docs = include_root_docs

    def discover_documents(self) -> Iterator[WikiDocument]:
        """
        Yields all indexable documents from configured paths and Confluence exports.
        """
        seen_paths = set()

        # 1. Confluence export if specified
        if self.confluence_export_path and self.confluence_export_path.exists():
            for doc in self._read_confluence_export(self.confluence_export_path):
                if doc.abs_path not in seen_paths:
                    seen_paths.add(doc.abs_path)
                    yield doc

        # 2. Custom wiki path if specified
        if self.wiki_path and self.wiki_path.exists():
            if self.wiki_path.is_file() and self.wiki_path.suffix.lower() in [".md", ".markdown", ".html"]:
                doc = self._read_doc(self.wiki_path)
                if doc and doc.abs_path not in seen_paths:
                    seen_paths.add(doc.abs_path)
                    yield doc
            elif self.wiki_path.is_dir():
                for root_dir, _, files in os.walk(self.wiki_path):
                    for fname in files:
                        p = Path(root_dir) / fname
                        if p.suffix.lower() in [".md", ".markdown", ".html"]:
                            doc = self._read_doc(p)
                            if doc and doc.abs_path not in seen_paths:
                                seen_paths.add(doc.abs_path)
                                yield doc

        # 3. Well-known doc directories inside repo_root
        for dir_name in WELL_KNOWN_DOC_DIRS:
            target_dir = self.repo_root / dir_name
            if target_dir.exists() and target_dir.is_dir():
                for root_dir, _, files in os.walk(target_dir):
                    for fname in files:
                        p = Path(root_dir) / fname
                        if p.suffix.lower() in [".md", ".markdown"]:
                            doc = self._read_doc(p)
                            if doc and doc.abs_path not in seen_paths:
                                seen_paths.add(doc.abs_path)
                                yield doc

        # 4. Well-known root documents
        if self.include_root_docs:
            for fname in WELL_KNOWN_DOC_FILES:
                p = self.repo_root / fname
                if p.exists() and p.is_file():
                    doc = self._read_doc(p)
                    if doc and doc.abs_path not in seen_paths:
                        seen_paths.add(doc.abs_path)
                        yield doc

    def _read_confluence_export(self, export_path: Path) -> Iterator[WikiDocument]:
        """Reads Confluence exported directory or zip archive."""
        if export_path.is_file() and export_path.suffix.lower() == ".zip":
            try:
                with zipfile.ZipFile(export_path, "r") as z:
                    for name in z.namelist():
                        if name.endswith((".html", ".md", ".markdown")) and not name.startswith("__MACOSX"):
                            content_bytes = z.read(name)
                            text = content_bytes.decode("utf-8", errors="replace")
                            if name.endswith(".html"):
                                title, content = convert_html_to_markdown(text)
                            else:
                                content = text
                                title = extract_title(content) or Path(name).stem.title()
                            if content:
                                yield WikiDocument(
                                    file_path=f"confluence/{name}",
                                    abs_path=f"{export_path}:{name}",
                                    title=title or Path(name).stem.title(),
                                    content=content,
                                    content_hash=hashlib.sha256(content.encode("utf-8")).hexdigest(),
                                    source_type="wiki",
                                )
            except Exception as e:
                logger.warning(f"Error reading Confluence zip export {export_path}: {e}")
        elif export_path.is_dir():
            for root_dir, _, files in os.walk(export_path):
                for fname in files:
                    p = Path(root_dir) / fname
                    if p.suffix.lower() in [".html", ".md", ".markdown"]:
                        doc = self._read_doc(p)
                        if doc:
                            yield doc

    def _read_doc(self, file_path: Path) -> Optional[WikiDocument]:
        """Read a single markdown or Confluence HTML file into a WikiDocument."""
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()

            rel_path = normalize_rel_path(file_path, self.repo_root)
            fallback_title = file_path.stem.replace("_", " ").replace("-", " ").title()

            if file_path.suffix.lower() == ".html":
                title, content = convert_html_to_markdown(raw)
                title = title or fallback_title
            else:
                content = raw
                title = extract_title(content, fallback_title)

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

            return WikiDocument(
                file_path=rel_path,
                abs_path=str(file_path),
                title=title,
                content=content,
                content_hash=content_hash,
                source_type="wiki",
            )
        except Exception:
            return None

    def get_all_documents(self) -> List[WikiDocument]:
        """Convenience method returning list of all discovered documents."""
        return list(self.discover_documents())


