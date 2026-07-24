"""Graph builder: constructs and queries knowledge graphs using NetworkX.

For future Graph RAG experiments. Currently a stub.
"""

from __future__ import annotations

import logging
from typing import Any

import networkx as nx

from my_agent.config.schema import GraphConfig
from my_agent.core.base import Document, RetrievedDocument

logger = logging.getLogger(__name__)


class NetworkXGraphBuilder:
    """Lightweight knowledge graph using NetworkX for Graph RAG experiments.

    Builds a graph from documents: each document is a node,
    edges represent co-occurrence or semantic similarity relationships.
    """

    def __init__(self, config: GraphConfig):
        self.config = config
        self.graph = nx.Graph()

    def build_from_documents(self, documents: list[Document]) -> None:
        """Build a simple co-occurrence graph from documents.

        Each document becomes a node. Edges are added between documents
        that share significant token overlap.
        """
        if len(documents) > self.config.max_nodes:
            documents = documents[: self.config.max_nodes]

        # Add nodes
        for doc in documents:
            self.graph.add_node(doc.id, content=doc.content, metadata=doc.metadata)

        # Add edges based on token overlap (simple heuristic)
        for i in range(len(documents)):
            tokens_i = set(documents[i].content.lower().split())
            for j in range(i + 1, len(documents)):
                tokens_j = set(documents[j].content.lower().split())
                overlap = len(tokens_i & tokens_j) / max(len(tokens_i | tokens_j), 1)
                if overlap > 0.1:
                    self.graph.add_edge(documents[i].id, documents[j].id, weight=overlap)

        logger.info(f"Built graph with {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")

    def traverse(self, start_nodes: list[str], max_hops: int | None = None) -> list[str]:
        """Get all nodes reachable from start_nodes within max_hops.

        Args:
            start_nodes: Node IDs to start traversal from
            max_hops: Maximum number of hops (defaults to config value)

        Returns:
            List of reachable node IDs
        """
        hops = max_hops if max_hops is not None else self.config.max_hops
        visited = set(start_nodes)

        frontier = set(start_nodes)
        for _ in range(hops):
            next_frontier = set()
            for node in frontier:
                if node in self.graph:
                    next_frontier.update(self.graph.neighbors(node))
            frontier = next_frontier - visited
            visited.update(frontier)
            if not frontier:
                break

        return list(visited)

    def clear(self) -> None:
        self.graph.clear()
