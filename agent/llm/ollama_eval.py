"""
agent/llm/ollama_eval.py

Calls a local Ollama model to evaluate whether a ChangedCodeUnit + its
Candidate matches represent duplicated logic or architectural violations.

The model is prompted to return STRICT JSON matching the Finding fields.
This module handles the HTTP call, JSON parsing, and validation.
"""

import json
import logging
from typing import Any, Dict, List, Optional

import requests

from shared.config import EVAL_MODEL
from shared.schemas import Candidate, ChangedCodeUnit

logger = logging.getLogger("agent.llm.ollama_eval")

OLLAMA_GENERATE_URL = "http://localhost:11434/api/generate"

_SYSTEM_PROMPT = """\
You are a code review assistant that detects duplicated logic and \
architectural convention violations. You will be given a NEW code unit \
from a pull request and one or more EXISTING code snippets from the \
codebase (or wiki style-guide rules) that are semantically similar.

Your job:
1. Determine if the new code is duplicating logic already present in \
   the existing snippets (is_duplicate).
2. Determine if the new code violates an architectural convention \
   described in a wiki/style-guide snippet (is_architectural_violation).
3. Assign a confidence score (0.0 to 1.0).
4. Provide concise reasoning (max 2 sentences).
5. If applicable, provide a suggested_replacement code snippet.

Respond with ONLY a valid JSON object, no markdown fencing, no extra text.
"""

_USER_PROMPT_TEMPLATE = """\
NEW CODE UNIT (from the PR):
  File: {file_path}
  Symbol: {symbol_name}
  Language: {language}
  Code:
```
{code}
```

EXISTING CANDIDATES (semantically similar snippets from the codebase/wiki):
{candidates_text}

Respond with ONLY this JSON structure:
{{
  "is_duplicate": true/false,
  "is_architectural_violation": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "max 2 sentences",
  "suggested_replacement": "replacement code or null"
}}
"""


def _format_candidates(candidates: List[Candidate]) -> str:
    """Format candidates into a readable text block for the prompt."""
    parts = []
    for i, c in enumerate(candidates, 1):
        source_label = "WIKI RULE" if c.source_type == "wiki" else "CODEBASE"
        parts.append(
            f"  [{i}] ({source_label}) {c.file_path}::{c.symbol_name} "
            f"(similarity={c.similarity:.2f})\n"
            f"  ```\n  {c.code_snippet}\n  ```"
        )
    return "\n\n".join(parts)


def _build_prompt(unit: ChangedCodeUnit, candidates: List[Candidate]) -> str:
    """Build the full user prompt for the Ollama model."""
    return _USER_PROMPT_TEMPLATE.format(
        file_path=unit.file_path,
        symbol_name=unit.symbol_name,
        language=unit.language,
        code=unit.code,
        candidates_text=_format_candidates(candidates),
    )


def _parse_llm_response(raw_text: str) -> Dict[str, Any]:
    """
    Parse the LLM's response text as JSON.

    Handles common LLM quirks:
    - Strips markdown code fences if present
    - Strips leading/trailing whitespace
    """
    text = raw_text.strip()

    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return json.loads(text)


def evaluate_with_ollama(
    unit: ChangedCodeUnit,
    candidates: List[Candidate],
    model: str = EVAL_MODEL,
    ollama_url: str = OLLAMA_GENERATE_URL,
    timeout: int = 120,
) -> Optional[Dict[str, Any]]:
    """
    Call Ollama to evaluate a (unit, candidates) pair.

    Returns:
        Parsed JSON dict with Finding fields, or None if the call fails.
    """
    prompt = _build_prompt(unit, candidates)

    logger.info(
        "Calling Ollama (%s) for %s::%s with %d candidates",
        model,
        unit.file_path,
        unit.symbol_name,
        len(candidates),
    )

    try:
        response = requests.post(
            ollama_url,
            json={
                "model": model,
                "system": _SYSTEM_PROMPT,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.0,  # strict deterministic JSON
                    "seed": 42,          # force deterministic random seed
                    "num_predict": 512,
                },
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        logger.error("Ollama request failed for %s::%s: %s", unit.file_path, unit.symbol_name, e)
        return None

    raw_response = response.json().get("response", "")
    print(f"\n--- RAW OLLAMA RESPONSE ---\n{raw_response}\n---------------------------\n")
    logger.debug("Raw Ollama response: %s", raw_response)

    try:
        parsed = _parse_llm_response(raw_response)
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(
            "Failed to parse Ollama JSON for %s::%s: %s\nRaw: %s",
            unit.file_path,
            unit.symbol_name,
            e,
            raw_response[:500],
        )
        return None

    # Validate required fields
    required = {"is_duplicate", "is_architectural_violation", "confidence", "reasoning"}
    missing = required - set(parsed.keys())
    if missing:
        logger.error(
            "Ollama response missing fields %s for %s::%s",
            missing,
            unit.file_path,
            unit.symbol_name,
        )
        return None

    return parsed


# ---------------------------------------------------------------------------
# Fake evaluator for testing when Ollama is not available
# ---------------------------------------------------------------------------

def fake_evaluate(
    unit: ChangedCodeUnit,
    candidates: List[Candidate],
) -> Dict[str, Any]:
    """
    Returns a deterministic fake evaluation result for testing.
    Always reports a duplicate finding with high confidence.
    """
    return {
        "is_duplicate": True,
        "is_architectural_violation": False,
        "confidence": 0.88,
        "reasoning": (
            f"The function '{unit.symbol_name}' in '{unit.file_path}' appears to "
            f"duplicate logic found in '{candidates[0].file_path}::{candidates[0].symbol_name}'."
        ),
        "suggested_replacement": (
            f"import {{ {candidates[0].symbol_name} }} from "
            f"'{candidates[0].file_path.replace('.ts', '')}';"
        ),
    }
