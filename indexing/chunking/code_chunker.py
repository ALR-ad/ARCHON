"""
indexing/chunking/code_chunker.py

PERSON A: tree-sitter based code chunking using tree_sitter_language_pack.
Input:  raw file text + language
Output: List[ChangedCodeUnit] (whole functions/classes/methods with context)
"""

import hashlib
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path

import tree_sitter_language_pack as tslp
from shared.schemas import ChangedCodeUnit

# Language alias mapping to match tree_sitter_language_pack identifiers
TS_LANGUAGE_MAP: Dict[str, str] = {
    "python": "python",
    "py": "python",
    "typescript": "typescript",
    "ts": "typescript",
    "tsx": "tsx",
    "javascript": "javascript",
    "js": "javascript",
    "jsx": "javascript",
    "go": "go",
    "java": "java",
    "rust": "rust",
    "rs": "rust",
    "c": "c",
    "cpp": "cpp",
    "c++": "cpp",
    "c_sharp": "c_sharp",
    "csharp": "c_sharp",
    "cs": "c_sharp",
    "ruby": "ruby",
    "rb": "ruby",
    "php": "php",
}

# Node types that represent function or class definitions across languages
FUNCTION_NODE_TYPES: Set[str] = {
    "function_definition",          # Python, C, C++
    "async_function_definition",    # Python
    "function_declaration",         # TS, JS, Go
    "method_declaration",           # Java, Go
    "method_definition",            # TS, JS
    "function_item",                # Rust
    "arrow_function",               # TS, JS
}

CLASS_NODE_TYPES: Set[str] = {
    "class_definition",             # Python, Ruby
    "class_declaration",            # TS, JS, Java, C#
    "interface_declaration",        # TS, Java, Go
    "type_declaration",             # Go
    "struct_item",                  # Rust
    "impl_item",                    # Rust
    "enum_declaration",             # TS, Java, C#
}


def compute_unit_id(file_path: str, symbol_name: str, code: str, commit_sha: str = "") -> str:
    """Generate a deterministic, stable hash for a code unit."""
    content = f"{file_path}:{symbol_name}:{commit_sha}:{code}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]


def get_node_text(node, source_bytes: bytes) -> str:
    """Extract decoded text for an AST node."""
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def get_preceding_comment(node, source_lines: List[str]) -> Tuple[str, int]:
    """
    Look for contiguous comment lines immediately preceding the node.
    Returns (comment_text, start_line_index).
    """
    node_start_line = node.start_point.row  # 0-indexed
    curr_line = node_start_line - 1
    comment_lines = []

    while curr_line >= 0:
        line = source_lines[curr_line].strip()
        if (
            line.startswith("#")
            or line.startswith("//")
            or line.startswith("/*")
            or line.startswith("*")
            or line.endswith("*/")
        ):
            comment_lines.insert(0, source_lines[curr_line])
            curr_line -= 1
        elif line == "":
            # Empty line between comments and code: stop collecting
            break
        else:
            break

    if comment_lines:
        return "\n".join(comment_lines) + "\n", curr_line + 1
    return "", node_start_line


def extract_symbol_name(node, source_bytes: bytes, parent_symbol: Optional[str] = None) -> str:
    """Extract symbol name from AST node, including field 'name' or declarator."""
    name_node = node.child_by_field_name("name")
    if name_node:
        name = get_node_text(name_node, source_bytes).strip()
        if parent_symbol:
            return f"{parent_symbol}.{name}"
        return name

    # For arrow functions or expressions assigned to a variable: const foo = () => ...
    if node.type == "arrow_function" and node.parent:
        if node.parent.type == "variable_declarator":
            p_name = node.parent.child_by_field_name("name")
            if p_name:
                name = get_node_text(p_name, source_bytes).strip()
                return f"{parent_symbol}.{name}" if parent_symbol else name

    # Check for type_identifier or identifier in children
    for child in node.children:
        if child.type in ("identifier", "type_identifier", "property_identifier"):
            name = get_node_text(child, source_bytes).strip()
            return f"{parent_symbol}.{name}" if parent_symbol else name

    return parent_symbol or "<anonymous>"


class CodeChunker:
    """
    Extracts semantic functions, methods, and classes using tree-sitter.
    """

    def __init__(self, language: str):
        normalized_lang = TS_LANGUAGE_MAP.get(language.lower(), language.lower())
        self.language_name = normalized_lang
        try:
            self.parser = tslp.get_parser(normalized_lang)
        except Exception:
            self.parser = None

    def chunk(
        self,
        code: str,
        file_path: str,
        commit_sha: str = "",
        diff_type: str = "added",
    ) -> List[ChangedCodeUnit]:
        """
        Parse code and extract logical units (functions, methods, classes).
        """
        if not code or not code.strip():
            return []

        if not self.parser:
            # Fallback if parser is not available for this language
            return self._fallback_whole_file(code, file_path, commit_sha, diff_type)

        source_bytes = code.encode("utf-8")
        try:
            tree = self.parser.parse(source_bytes)
        except Exception:
            return self._fallback_whole_file(code, file_path, commit_sha, diff_type)

        source_lines = code.splitlines()
        units: List[ChangedCodeUnit] = []
        seen_ranges: Set[Tuple[int, int]] = set()

        def visit(node, parent_symbol: Optional[str] = None):
            # If parent was export_statement, skip unless it's a class where we visit methods
            if node.parent and node.parent.type == "export_statement":
                if node.type in CLASS_NODE_TYPES:
                    class_name = extract_symbol_name(node, source_bytes, parent_symbol)
                    for child in node.children:
                        visit(child, parent_symbol=class_name)
                return

            target_node = node
            decl_node = node

            # Check if this node is an export_statement wrapping a function/class/declaration
            if node.type == "export_statement":
                found_decl = False
                for child in node.children:
                    if child.type in FUNCTION_NODE_TYPES or child.type in CLASS_NODE_TYPES:
                        decl_node = child
                        found_decl = True
                        break
                    if child.type in ("lexical_declaration", "variable_declaration"):
                        for decl in child.children:
                            if decl.type == "variable_declarator":
                                val = decl.child_by_field_name("value")
                                if val and val.type in FUNCTION_NODE_TYPES:
                                    decl_node = decl
                                    found_decl = True
                                    break
                if not found_decl:
                    for child in node.children:
                        visit(child, parent_symbol=parent_symbol)
                    return
                target_node = node

            is_arrow_var = False
            if decl_node.type == "variable_declarator":
                val = decl_node.child_by_field_name("value")
                if val and val.type in FUNCTION_NODE_TYPES:
                    is_arrow_var = True

            node_type = decl_node.type
            if node_type in FUNCTION_NODE_TYPES or node_type in CLASS_NODE_TYPES or is_arrow_var:
                start_row = target_node.start_point.row
                end_row = target_node.end_point.row

                # Check if already processed
                if (start_row, end_row) not in seen_ranges:
                    seen_ranges.add((start_row, end_row))

                    symbol_name = extract_symbol_name(decl_node, source_bytes, parent_symbol)
                    node_text = get_node_text(target_node, source_bytes)

                    # Prepend preceding comments/docstrings if present
                    comment_prefix, comment_start = get_preceding_comment(target_node, source_lines)
                    full_code = comment_prefix + node_text
                    start_line = comment_start + 1
                    end_line = end_row + 1

                    unit_id = compute_unit_id(file_path, symbol_name, full_code, commit_sha)

                    units.append(
                        ChangedCodeUnit(
                            unit_id=unit_id,
                            file_path=file_path,
                            symbol_name=symbol_name,
                            language=self.language_name,
                            start_line=start_line,
                            end_line=end_line,
                            code=full_code,
                            diff_type="added" if diff_type == "added" else "modified",
                        )
                    )

                # For classes, also visit internal methods
                if node_type in CLASS_NODE_TYPES:
                    class_name = extract_symbol_name(decl_node, source_bytes, parent_symbol)
                    for child in decl_node.children:
                        visit(child, parent_symbol=class_name)
                    return

            # Continue recursing
            for child in node.children:
                visit(child, parent_symbol=parent_symbol)

        visit(tree.root_node)

        # If no units were extracted (e.g. script with top-level procedural statements),
        # treat the entire file as a code unit so it is indexable.
        if not units:
            units = self._fallback_whole_file(code, file_path, commit_sha, diff_type)

        return units

    def _fallback_whole_file(
        self,
        code: str,
        file_path: str,
        commit_sha: str = "",
        diff_type: str = "added",
    ) -> List[ChangedCodeUnit]:
        """Fallback when AST chunking cannot extract distinct functions/classes."""
        lines = code.splitlines()
        symbol_name = Path(file_path).stem or "<module>"
        unit_id = compute_unit_id(file_path, symbol_name, code, commit_sha)
        return [
            ChangedCodeUnit(
                unit_id=unit_id,
                file_path=file_path,
                symbol_name=symbol_name,
                language=self.language_name,
                start_line=1,
                end_line=len(lines) if lines else 1,
                code=code,
                diff_type="added" if diff_type == "added" else "modified",
            )
        ]


def chunk_code(
    file_path: str,
    code: str,
    language: str,
    commit_sha: str = "",
    diff_type: str = "added",
) -> List[ChangedCodeUnit]:
    """
    Functional helper to chunk source code into whole function/class units.
    """
    chunker = CodeChunker(language)
    return chunker.chunk(code=code, file_path=file_path, commit_sha=commit_sha, diff_type=diff_type)

