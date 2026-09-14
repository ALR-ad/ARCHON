import pytest
from indexing.chunking.code_chunker import chunk_code, CodeChunker
from shared.schemas import ChangedCodeUnit


def test_chunk_python_functions_and_classes():
    py_code = """
# Utility comment
def add(a: int, b: int) -> int:
    \"\"\"Adds two numbers.\"\"\"
    return a + b

class Calculator:
    \"\"\"Simple calculator.\"\"\"
    def multiply(self, a: int, b: int) -> int:
        return a * b
"""
    units = chunk_code("src/calc.py", py_code, "python")
    symbols = {u.symbol_name: u for u in units}

    assert "add" in symbols
    assert "Calculator" in symbols
    assert "Calculator.multiply" in symbols

    # Check that ChangedCodeUnit contract is satisfied
    for u in units:
        assert isinstance(u, ChangedCodeUnit)
        assert u.file_path == "src/calc.py"
        assert u.language == "python"
        assert u.start_line > 0
        assert u.end_line >= u.start_line
        assert u.diff_type in ("added", "modified")

    # Check that docstring and preceding comments are included
    add_unit = symbols["add"]
    assert "Utility comment" in add_unit.code
    assert "def add(a: int, b: int) -> int:" in add_unit.code
    assert "Adds two numbers" in add_unit.code

    calc_unit = symbols["Calculator"]
    assert "class Calculator:" in calc_unit.code


def test_chunk_typescript_exports_and_methods():
    ts_code = """
import { DateUtil } from './date';

// Parses order date with ISO or legacy format
export function parseOrderDate(raw: string): Date {
    return new Date(raw);
}

export class OrderService {
    calculateTotal(items: number[]): number {
        return items.reduce((a, b) => a + b, 0);
    }
}

export const formatDate = (d: Date): string => {
    return d.toISOString();
};
"""
    units = chunk_code("src/orders/parseOrderDate.ts", ts_code, "typescript")
    symbols = {u.symbol_name: u for u in units}

    assert "parseOrderDate" in symbols
    assert "OrderService" in symbols
    assert "OrderService.calculateTotal" in symbols
    assert "formatDate" in symbols

    # Check comments preserved
    parse_unit = symbols["parseOrderDate"]
    assert "Parses order date" in parse_unit.code
    assert "export function parseOrderDate" in parse_unit.code


def test_chunker_fallback_for_procedural_script():
    script_code = """
import os
import sys

print("Running setup step 1")
print("Running setup step 2")
"""
    units = chunk_code("scripts/setup.py", script_code, "python")
    assert len(units) >= 1
    assert units[0].file_path == "scripts/setup.py"
    assert "setup step 1" in units[0].code
