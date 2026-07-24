"""LangGraph adapter: maps RAGPipeline to LangGraph's StateGraph.

This is a thin wrapper (~50 lines) that exposes core pipeline steps as LangGraph nodes.
"""

from __future__ import annotations

from typing import TypedDict

from my_agent.core.pipeline import PipelineContext, RAGPipeline

try:
    from langgraph.graph import StateGraph, END
    HAS_LANGGRAPH = True
except ImportError:
    HAS_LANGGRAPH = False


class RAGState(TypedDict):
    """LangGraph state shared across nodes."""

    query: str
    retrieved_docs: list
    answer: str
    done: bool


def build_rag_graph(pipeline: RAGPipeline) -> "StateGraph":
    """Build a LangGraph StateGraph from a RAGPipeline.

    Graph structure:
        START → retrieve → synthesize → END

    Args:
        pipeline: An initialized RAGPipeline instance

    Returns:
        A compiled LangGraph StateGraph ready for invocation
    """
    if not HAS_LANGGRAPH:
        raise ImportError("langgraph is not installed. Run: pip install langgraph>=0.2.0")

    def retrieve_node(state: RAGState) -> dict:
        docs = pipeline.retriever.retrieve(state["query"], top_k=pipeline.config.retriever.top_k)
        return {"retrieved_docs": docs}

    def synthesize_node(state: RAGState) -> dict:
        template = pipeline.config.synthesizer.prompt_template
        answer = pipeline.synthesizer.synthesize(state["query"], state["retrieved_docs"], prompt_template=template)
        return {"answer": answer, "done": True}

    graph = StateGraph(RAGState)

    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", END)

    return graph.compile()


def invoke_rag_graph(pipeline: RAGPipeline, query: str) -> dict:
    """Shortcut: build graph and invoke in one call."""
    graph = build_rag_graph(pipeline)
    return graph.invoke({"query": query, "retrieved_docs": [], "answer": "", "done": False})
