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

from indexing.config import OLLAMA_HOST
from shared.config import EVAL_MODEL
from shared.schemas import Candidate, ChangedCodeUnit

logger = logging.getLogger("agent.llm.ollama_eval")

OLLAMA_GENERATE_URL = f"{OLLAMA_HOST.rstrip('/')}/api/generate"

# Shared session keeps the connection alive between calls (a fresh connection to
# "localhost" on Windows tries IPv6 first and adds ~2s per call).
_session = requests.Session()

_SYSTEM_PROMPT = """\
You are a code review assistant. You will be given a NEW code unit from a \
pull request and a numbered list of EXISTING candidates. Each candidate is \
either a CODEBASE snippet or a WIKI RULE. Answer two separate questions.

1. is_duplicate -- judged ONLY against CODEBASE candidates.
   True only if the new code re-implements the same logic as a CODEBASE \
snippet, so it could call or reuse that snippet instead. Similar structure, \
shared boilerplate, or the same topic is NOT duplication. If there are no \
CODEBASE candidates, is_duplicate must be false.

2. is_architectural_violation -- judged ONLY against WIKI RULE candidates.
   True only if a WIKI RULE states a requirement or a prohibition AND the \
new code itself clearly does what the rule forbids, or clearly fails to do \
what it requires. It is NOT a violation if the rule is about something the \
code does not do, if the code complies with the rule, or if the rule merely \
describes the project. Most code breaks no rule: when in doubt, answer false. \
If true, set violated_rule to the number of that WIKI RULE; otherwise null. \
If there are no WIKI RULE candidates, is_architectural_violation must be false.

Also give a confidence score (0.0 to 1.0), concise reasoning (max 2 \
sentences) naming the snippet or rule you relied on, and, if applicable, a \
suggested_replacement code snippet.

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

EXISTING CANDIDATES (CODEBASE snippets and WIKI RULES):
{candidates_text}

Respond with ONLY this JSON structure:
{{
  "is_duplicate": true/false,
  "is_architectural_violation": true/false,
  "violated_rule": number of the broken WIKI RULE or null,
  "confidence": 0.0-1.0,
  "reasoning": "max 2 sentences",
  "suggested_replacement": "replacement code or null"
}}
"""


def _format_candidates(candidates: List[Candidate]) -> str:
    """Format candidates into a readable text block for the prompt."""
    parts = []
    for i, c in enumerate(candidates, 1):
        if c.source_type == "wiki":
            # No similarity score for rules: it says nothing about whether one is broken
            parts.append(f"  [{i}] (WIKI RULE) {c.file_path}\n  {c.code_snippet}")
            continue
        parts.append(
            f"  [{i}] (CODEBASE) {c.file_path}::{c.symbol_name} "
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
        response = _session.post(
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
# Second opinion on rule violations
# ---------------------------------------------------------------------------

_CONFIRM_SYSTEM_PROMPT = """\
You check one piece of code against one written rule and decide whether the \
code breaks it.

First work out exactly what the rule forbids or requires. Then look for that \
exact thing in the code. Code that uses the approved mechanism the rule names \
(for example, it calls the repository, wrapper or helper the rule tells it to \
use) complies with the rule. Code that has nothing to do with the rule also \
complies.

Respond with ONLY a valid JSON object:
{
  "offending_code": "the exact line from the code that breaks the rule, copied character for character, or an empty string if no line does",
  "violates": true/false,
  "reasoning": "max 2 sentences"
}
"""

_CONFIRM_PROMPT_TEMPLATE = """\
RULE:
{rule}

CODE ({language}, {file_path}):
```
{code}
```
"""


def _squash(text: str) -> str:
    """Drop whitespace and unify quote characters, so a line the model re-typed still matches."""
    return "".join(text.split()).replace("'", '"')


def _quote_appears_in(quote: str, code: str) -> bool:
    """True if at least one quoted line (4+ characters) really occurs in the code."""
    haystack = _squash(code)
    lines = [_squash(line) for line in quote.splitlines()]
    return any(len(line) >= 4 and line in haystack for line in lines)


def confirm_violation_with_ollama(
    unit: ChangedCodeUnit,
    rule: Candidate,
    model: str = EVAL_MODEL,
    ollama_url: str = OLLAMA_GENERATE_URL,
    timeout: int = 120,
) -> bool:
    """
    Re-check a flagged rule violation with a narrower question: this one rule,
    this one unit, and the model must quote the offending line. The violation
    stands only if it says yes and the quote really appears in the code.

    Returns False if the call fails, so an unverified violation is not reported.
    """
    try:
        response = _session.post(
            ollama_url,
            json={
                "model": model,
                "system": _CONFIRM_SYSTEM_PROMPT,
                "prompt": _CONFIRM_PROMPT_TEMPLATE.format(
                    rule=rule.code_snippet,
                    language=unit.language,
                    file_path=unit.file_path,
                    code=unit.code,
                ),
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0, "seed": 42, "num_predict": 256},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        parsed = _parse_llm_response(response.json().get("response", ""))
    except (requests.RequestException, json.JSONDecodeError, ValueError) as e:
        logger.error("Violation check failed for %s::%s: %s", unit.file_path, unit.symbol_name, e)
        return False

    confirmed = bool(parsed.get("violates")) and _quote_appears_in(
        str(parsed.get("offending_code") or ""), unit.code
    )
    logger.info(
        "Violation check for %s::%s: %s (%s)",
        unit.file_path,
        unit.symbol_name,
        "confirmed" if confirmed else "rejected",
        str(parsed.get("reasoning", ""))[:160],
    )
    return confirmed


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
