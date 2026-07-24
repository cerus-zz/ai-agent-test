"""Test adapters in isolation."""


def test_langgraph_adapter_registers():
    """Verify the adapter module loads without import errors."""
    try:
        from my_agent.adapters.langgraph_adapter import build_rag_graph, invoke_rag_graph
        assert callable(build_rag_graph), "build_rag_graph should be callable"
    except ImportError as e:
        # LangGraph might not be installed, that's OK for this test
        import pytest
        pytest.skip(f"LangGraph not installed: {e}")
