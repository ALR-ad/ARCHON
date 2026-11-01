"""
Tests for agent/review_filters.py

Verifies:
1. Test files are ignored by default; review_ignore_paths overrides the defaults
2. references_symbol() catches calls, imports and pass-as-value, not prose
3. is_caller_or_wrapper() works in both directions
"""

from agent.review_filters import (
    is_caller_or_wrapper,
    is_review_ignored,
    references_symbol,
)


def test_default_ignores_test_files():
    for path in [
        "tests/test_embedder.py",
        "tests/agent/test_webhook.py",
        "src/foo/test_bar.py",
        "pkg/server_test.go",
        "web/src/app.spec.ts",
        "web/src/__tests__/app.tsx",
        "conftest.py",
    ]:
        assert is_review_ignored(path), path


def test_default_keeps_source_files():
    for path in ["agent/dag.py", "src/latest/parse.ts", "indexing/embeddings/embedder.py"]:
        assert not is_review_ignored(path), path


def test_custom_patterns_replace_defaults():
    assert not is_review_ignored("tests/test_x.py", [])
    assert is_review_ignored("db/migrations/0001.py", ["**/migrations/**"])


def test_references_symbol_call():
    assert references_symbol("result = evaluate_with_ollama(unit, cands)", "evaluate_with_ollama")


def test_references_symbol_aliased_import():
    code = "def embed_text(text):\n    from indexing.embeddings.embedder import embed_text as _impl\n    return _impl(text)"
    assert references_symbol(code, "embed_text")


def test_references_symbol_passed_as_value():
    assert references_symbol("if eval_fn is None:\n    eval_fn = evaluate_with_ollama", "evaluate_with_ollama")


def test_references_symbol_ignores_own_definition_and_prose():
    code = 'def parse_date(raw):\n    """We parse_date here, see parse docs."""\n    return raw'
    assert not references_symbol(code, "parse_date")


def test_references_symbol_method_name():
    assert references_symbol("x = self.store.query(vec)", "QdrantVectorStore.query")


def test_references_symbol_ignores_short_names():
    assert not references_symbol("f(x)", "f")


def test_is_caller_or_wrapper_both_directions():
    caller = "def evaluate(matches):\n    return [evaluate_with_ollama(u, c) for u, c in matches]"
    callee = "def evaluate_with_ollama(unit, candidates):\n    return post(unit)"
    assert is_caller_or_wrapper(callee, "evaluate_with_ollama", caller, "evaluate")
    assert is_caller_or_wrapper(caller, "evaluate", callee, "evaluate_with_ollama")


def test_unrelated_duplicates_are_not_callers():
    a = "def parse_order_date(raw):\n    return datetime.strptime(raw, '%Y-%m-%d')"
    b = "def parse_date(s):\n    return datetime.strptime(s, '%Y-%m-%d')"
    assert not is_caller_or_wrapper(a, "parse_order_date", b, "parse_date")
