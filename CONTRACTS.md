# Contracts

The full contract lives in code, not prose, so it can't drift:
- `shared/schemas.py`    -- ChangedCodeUnit, EmbeddedUnit, Candidate, Finding
- `shared/interfaces.py` -- embed_text(), VectorStore.query(), VectorStore.get_index_status()

Person B uses `agent/llm/mock_vectorstore.py` (matches the same interface)
until Milestone 2, when Person A's real `indexing/vectorstore/store.py` is swapped in.

Any change to shared/schemas.py or shared/interfaces.py requires review from BOTH people.
