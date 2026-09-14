import pytest
from indexing.chunking.wiki_chunker import chunk_wiki, WikiChunk


def test_wiki_heading_and_atomic_rule_chunking():
    wiki_content = """# Architecture Guidelines

## Database Access
- Rule: Controllers must not call the database layer directly.
- Rule: All database queries must be encapsulated inside repository classes.

## Service Layer
> Rule: Business logic belongs in domain services, not controllers.
Controllers should only handle request validation and HTTP response formatting.
"""
    chunks = chunk_wiki(wiki_content, "docs/ARCHITECTURE.md")

    assert len(chunks) >= 3
    snippets = [c.code_snippet for c in chunks]

    # Verify atomic rules are captured
    assert any("Controllers must not call the database layer directly" in s for s in snippets)
    assert any("All database queries must be encapsulated inside repository" in s for s in snippets)
    assert any("Business logic belongs in domain services" in s for s in snippets)

    # Verify heading hierarchy context
    for c in chunks:
        assert isinstance(c, WikiChunk)
        assert c.source_type == "wiki"
        assert c.symbol_name == ""  # Candidate schema requirement for wiki
        assert c.file_path == "docs/ARCHITECTURE.md"
        assert "[" in c.code_snippet and "]" in c.code_snippet


def test_wiki_chunker_empty_and_fallback():
    assert chunk_wiki("", "docs/empty.md") == []

    simple = "Just a single paragraph without any headings or rules."
    chunks = chunk_wiki(simple, "docs/simple.md")
    assert len(chunks) == 1
    assert "Just a single paragraph" in chunks[0].code_snippet
