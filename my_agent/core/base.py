"""Abstract base classes for all core components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A document chunk with its embedding and metadata."""

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None


@dataclass
class RetrievedDocument(Document):
    """A document returned by a retriever, with a similarity score."""

    score: float = 0.0


@dataclass
class QueryResult:
    """Complete result of a RAG query."""

    query: str
    answer: str
    retrieved_docs: list[RetrievedDocument] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseRetriever(ABC):
    """Abstract retriever: indexes documents and retrieves relevant ones."""

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Index a list of documents into the vector store."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        """Retrieve top-k documents relevant to the query."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all indexed documents."""
        ...


class BaseSynthesizer(ABC):
    """Abstract synthesizer: generates an answer from retrieved documents and a query."""

    @abstractmethod
    def synthesize(self, query: str, documents: list[RetrievedDocument]) -> str:
        """Generate an answer based on retrieved documents."""
        ...


class BaseMemory(ABC):
    """Abstract memory: stores and retrieves conversation context."""

    @abstractmethod
    def add(self, key: str, value: Any) -> None:
        """Store a value."""
        ...

    @abstractmethod
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value."""
        ...

    @abstractmethod
    def clear(self) -> None:
        """Clear all memory."""
        ...
