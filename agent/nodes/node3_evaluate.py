"""
PERSON B: Node 3 (the only LLM node, runs via Ollama locally).
Input:  List[Tuple[ChangedCodeUnit, List[Candidate]]]
Output: List[Finding]  (see shared/schemas.py), filtered by CONFIDENCE_THRESHOLD
"""
