"""
indexing/chunking/wiki_chunker.py

PERSON A: heading-level chunking of wiki pages into atomic rules.
Breaks down architectural guides and style documents into focused,
single-rule chunks (e.g. "Rule: controllers must not call the DB directly")
with full heading path metadata.
"""

from dataclasses import dataclass
import hashlib
import re
from typing import List, Optional, Tuple


@dataclass
class WikiChunk:
    chunk_id: str
    file_path: str
    heading: str
    symbol_name: str          # Empty string for wiki chunks per Candidate schema
    code_snippet: str         # The rule / guideline text
    source_type: str          # Always "wiki"
    start_line: int
    end_line: int
    content_hash: str


def compute_wiki_chunk_id(file_path: str, heading: str, index: int, text: str) -> str:
    """Compute deterministic chunk ID for wiki rules."""
    raw = f"{file_path}:{heading}:{index}:{text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class WikiChunker:
    """
    Parses Markdown wiki/style-guide documents into heading sections and atomic rules.
    """

    def __init__(self, min_rule_chars: int = 15):
        self.min_rule_chars = min_rule_chars

    def chunk(self, content: str, file_path: str) -> List[WikiChunk]:
        """
        Extract semantic sections and atomic rules from markdown content.
        """
        if not content or not content.strip():
            return []

        lines = content.splitlines()
        chunks: List[WikiChunk] = []
        heading_stack: List[Tuple[int, str]] = []  # (level, heading_text)

        # Buffer lines for the current section
        current_section_lines: List[Tuple[int, str]] = [] # (1-based line_no, line_text)
        current_heading_path = ""

        def flush_current_section():
            nonlocal current_section_lines, current_heading_path
            if not current_section_lines:
                return

            section_chunks = self._chunk_section(
                file_path=file_path,
                heading_path=current_heading_path,
                lines=current_section_lines,
                chunk_index_offset=len(chunks),
            )
            chunks.extend(section_chunks)
            current_section_lines = []

        for line_idx, line in enumerate(lines):
            line_no = line_idx + 1
            stripped = line.strip()

            # Check for Markdown heading: #, ##, ###, etc.
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
            if heading_match:
                flush_current_section()
                level = len(heading_match.group(1))
                h_text = heading_match.group(2).strip()

                # Pop headings of equal or deeper level
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                heading_stack.append((level, h_text))
                current_heading_path = " > ".join(h[1] for h in heading_stack)
            else:
                current_section_lines.append((line_no, line))

        flush_current_section()

        # If no heading-based chunks could be formed, treat content as one chunk
        if not chunks:
            text = content.strip()
            if text:
                chunk_id = compute_wiki_chunk_id(file_path, "General", 0, text)
                chunks.append(
                    WikiChunk(
                        chunk_id=chunk_id,
                        file_path=file_path,
                        heading="General",
                        symbol_name="",
                        code_snippet=text,
                        source_type="wiki",
                        start_line=1,
                        end_line=len(lines),
                        content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    )
                )

        return chunks

    def _chunk_section(
        self,
        file_path: str,
        heading_path: str,
        lines: List[Tuple[int, str]],
        chunk_index_offset: int,
    ) -> List[WikiChunk]:
        """
        Split a section into atomic rules (bullet points, numbered lists, blockquotes)
        or paragraphs.
        """
        section_chunks: List[WikiChunk] = []
        heading_label = heading_path if heading_path else "General"

        # Separate into atomic items: list items, blockquotes, or paragraphs
        items: List[Tuple[int, int, str]] = []  # (start_line, end_line, item_text)
        current_item_lines: List[str] = []
        item_start_line = lines[0][0] if lines else 1

        for line_no, raw_line in lines:
            stripped = raw_line.strip()

            # Detect list item or blockquote start:
            # - ..., * ..., 1. ..., > ...
            is_bullet = bool(re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+\.\s+", stripped))
            is_quote = stripped.startswith(">")

            if is_bullet or is_quote:
                # Flush previous item if non-empty
                if current_item_lines:
                    text = "\n".join(current_item_lines).strip()
                    if len(text) >= self.min_rule_chars:
                        items.append((item_start_line, line_no - 1, text))
                    current_item_lines = []

                item_start_line = line_no
                # Strip list/quote prefix
                cleaned = re.sub(r"^[-*+]\s+", "", stripped)
                cleaned = re.sub(r"^\d+\.\s+", "", cleaned)
                cleaned = re.sub(r"^>\s*", "", cleaned)
                current_item_lines.append(cleaned)
            elif stripped == "":
                # Empty line: paragraph separator
                if current_item_lines:
                    text = "\n".join(current_item_lines).strip()
                    if len(text) >= self.min_rule_chars:
                        items.append((item_start_line, line_no - 1, text))
                    current_item_lines = []
                item_start_line = line_no + 1
            else:
                if not current_item_lines:
                    item_start_line = line_no
                current_item_lines.append(stripped)

        if current_item_lines:
            text = "\n".join(current_item_lines).strip()
            end_line = lines[-1][0] if lines else item_start_line
            if len(text) >= self.min_rule_chars:
                items.append((item_start_line, end_line, text))

        for idx, (start_l, end_l, item_text) in enumerate(items):
            # Prefix with heading context for optimal retrieval relevance
            full_snippet = f"[{heading_label}] {item_text}"
            c_id = compute_wiki_chunk_id(file_path, heading_label, chunk_index_offset + idx, full_snippet)
            content_hash = hashlib.sha256(full_snippet.encode("utf-8")).hexdigest()

            section_chunks.append(
                WikiChunk(
                    chunk_id=c_id,
                    file_path=file_path,
                    heading=heading_label,
                    symbol_name="",
                    code_snippet=full_snippet,
                    source_type="wiki",
                    start_line=start_l,
                    end_line=end_l,
                    content_hash=content_hash,
                )
            )

        return section_chunks


def chunk_wiki(content: str, file_path: str) -> List[WikiChunk]:
    """Functional convenience helper to chunk wiki content."""
    chunker = WikiChunker()
    return chunker.chunk(content=content, file_path=file_path)

