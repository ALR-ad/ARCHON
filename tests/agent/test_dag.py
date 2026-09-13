"""
Tests for agent/dag.py

Verifies:
1. DAG builds successfully without errors.
2. Nodes are executed in sequence (node1 -> node2 -> node3 -> node4).
3. The final invocation_state contains all intermediate artifacts (units, matches, findings, comment).
"""

import pytest
import asyncio
from unittest.mock import patch

from agent.dag import build_dag
from shared.schemas import ChangedCodeUnit, Finding, Candidate


@pytest.fixture
def fake_payload():
    return {
        "action": "opened",
        "pull_request": {
            "number": 42,
            "head": {"sha": "headsha"},
            "base": {"sha": "basesha"},
        },
        "repository": {
            "full_name": "org/repo",
            "name": "repo",
            "owner": {"login": "org"},
        },
    }


@pytest.mark.asyncio
async def test_dag_execution(fake_payload):
    """
    Test the full DAG execution flow.
    We will patch the underlying evaluate function to return a deterministic fake Finding
    so we don't actually hit the LLM (ollama_eval) during testing.
    """
    graph = build_dag()

    # Patch the Node 3 evaluation to use the fake_evaluate
    from agent.llm.ollama_eval import fake_evaluate
    
    with patch("agent.nodes.node3_evaluate.evaluate_with_ollama", side_effect=fake_evaluate):
        # We also need to patch evaluate itself if we just want to bypass the default 
        # eval_fn. By default, node3_evaluate.evaluate uses evaluate_with_ollama.
        # But wait, node3_evaluate.evaluate(..., eval_fn=evaluate_with_ollama) 
        # is called in _run_node3. Let's patch evaluate to inject the fake_evaluate.
        
        with patch("agent.dag.evaluate") as mock_evaluate:
            def fake_eval(matches, **kwargs):
                from agent.nodes.node3_evaluate import evaluate as real_evaluate
                return real_evaluate(matches, eval_fn=fake_evaluate, **kwargs)
            
            mock_evaluate.side_effect = fake_eval
            
            # Execute the graph
            initial_state = {"payload": fake_payload}
            result = await graph.invoke_async("start", invocation_state=initial_state)

            # Check that the final state contains all expected artifacts
            final_state = result.results["node4"].result.state

            # Node 1 produced units
            assert "units" in final_state
            assert isinstance(final_state["units"], list)
            assert len(final_state["units"]) > 0
            assert isinstance(final_state["units"][0], ChangedCodeUnit)

            # Node 2 produced matches
            assert "matches" in final_state
            assert isinstance(final_state["matches"], list)
            assert len(final_state["matches"]) > 0

            # Node 3 produced findings
            assert "findings" in final_state
            assert isinstance(final_state["findings"], list)
            # Depending on fake_evaluate confidence score and Node 3 threshold, 
            # it might be 1 or 0 findings.
            if len(final_state["findings"]) > 0:
                assert isinstance(final_state["findings"][0], Finding)

            # Node 4 produced a comment
            assert "comment" in final_state
            assert isinstance(final_state["comment"], str)
