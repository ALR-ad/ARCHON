"""
agent/dag.py

Wires Nodes 1-4 into a Strands Agents SDK Graph.
Uses a lightweight FunctionAgent adapter to wrap our deterministic
functions into the AgentBase interface expected by the graph.
"""

from typing import Any, Callable, Dict, Optional
import functools

from strands.agent.base import AgentBase
from strands.agent.agent_result import AgentResult
from strands.telemetry.metrics import EventLoopMetrics
from strands.multiagent.graph import GraphBuilder, Graph

# Import our node functions
from agent.nodes.node1_ingest_diff import ingest_diff
from agent.nodes.node2_semantic_search import semantic_search
from agent.nodes.node3_evaluate import evaluate
from agent.nodes.node4_comment import render_comment
from agent.llm.mock_vectorstore import MockVectorStore
from shared.interfaces import VectorStore


class FunctionAgent(AgentBase):
    """
    Lightweight adapter to make a deterministic function act as a Strands node.
    Reads from and writes to the graph's invocation_state dictionary.
    """

    def __init__(self, name: str, fn: Callable[[Dict[str, Any]], None]):
        self.name = name
        self.fn = fn

    def __call__(self, prompt=None, **kwargs):
        raise NotImplementedError("Use invoke_async/stream_async")

    async def invoke_async(self, prompt=None, **kwargs):
        raise NotImplementedError("Graph uses stream_async internally")

    async def stream_async(self, prompt=None, **kwargs):
        import asyncio
        state = kwargs.get("invocation_state", {})
        
        # Execute the underlying deterministic function in a thread to prevent blocking Uvicorn's event loop
        await asyncio.to_thread(self.fn, state)
        
        # Yield the required Strands event format
        res = AgentResult(
            message={"role": "assistant", "content": []},
            metrics=EventLoopMetrics(),
            stop_reason="end_turn",
            state=state,
        )
        yield {"result": res}


# -- Node Wrapper Functions --------------------------------------------------

def _run_node1(state: Dict[str, Any]) -> None:
    payload = state.get("payload", {})
    state["units"] = ingest_diff(payload)


def _run_node2(state: Dict[str, Any], store: VectorStore, embed_fn: Optional[Callable] = None) -> None:
    units = state.get("units", [])
    if embed_fn:
        state["matches"] = semantic_search(units, store, embed_fn=embed_fn)
    else:
        state["matches"] = semantic_search(units, store)


def _run_node3(state: Dict[str, Any], eval_fn: Optional[Callable] = None) -> None:
    matches = state.get("matches", [])
    if eval_fn:
        state["findings"] = evaluate(matches, eval_fn=eval_fn)
    else:
        state["findings"] = evaluate(matches)


def _run_node4(state: Dict[str, Any]) -> None:
    findings = state.get("findings", [])
    state["comment"] = render_comment(findings)


# -- DAG Builder -------------------------------------------------------------

def build_dag(
    store: Optional[VectorStore] = None,
    embed_fn: Optional[Callable] = None,
    eval_fn: Optional[Callable] = None,
) -> Graph:
    """
    Constructs the 4-node sequential DAG using Strands GraphBuilder.
    Dependencies can be injected for production, otherwise defaults to mocks.
    """
    if store is None:
        store = MockVectorStore()

    builder = GraphBuilder()

    # Create bound node functions
    node2_bound = functools.partial(_run_node2, store=store, embed_fn=embed_fn)
    node3_bound = functools.partial(_run_node3, eval_fn=eval_fn)

    # Add nodes
    builder.add_node(FunctionAgent("node1", _run_node1), "node1")
    builder.add_node(FunctionAgent("node2", node2_bound), "node2")
    builder.add_node(FunctionAgent("node3", node3_bound), "node3")
    builder.add_node(FunctionAgent("node4", _run_node4), "node4")

    # Wire them sequentially
    builder.add_edge("node1", "node2")
    builder.add_edge("node2", "node3")
    builder.add_edge("node3", "node4")

    builder.set_entry_point("node1")

    return builder.build()

