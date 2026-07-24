"""Configuration schema definitions using Pydantic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """OpenAI-compatible LLM configuration."""

    base_url: str = Field(default="https://api.openai.com/v1", description="API base URL")
    api_key: str = Field(default="sk-xxx", description="API key")
    model: str = Field(default="gpt-4o-mini", description="Model name")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=2048)


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model or OpenAI embedding model name",
    )
    provider: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    base_url: str | None = None  # for OpenAI-compatible embedding
    api_key: str | None = None


class ChromaConfig(BaseModel):
    """ChromaDB configuration."""

    persist_directory: str = Field(default="./chroma_db", description="ChromaDB persistence directory")
    collection_name: str = Field(default="rag_documents", description="ChromaDB collection name")


class RetrieverConfig(BaseModel):
    """Retriever configuration."""

    strategy: Literal["naive", "hyde", "graph"] = "naive"
    top_k: int = Field(default=5, description="Number of documents to retrieve")
    similarity_threshold: float | None = Field(default=None, description="Minimum similarity score")


class GraphConfig(BaseModel):
    """Graph RAG configuration (NetworkX)."""

    enabled: bool = False
    max_nodes: int = Field(default=1000, description="Max nodes in graph")
    max_hops: int = Field(default=2, description="Max hops for graph traversal")


class SynthesizerConfig(BaseModel):
    """Synthesizer (answer generation) configuration."""

    prompt_template: str | None = Field(
        default=None,
        description="Custom prompt template. If None, uses built-in template for the strategy.",
    )


class RAGConfig(BaseModel):
    """Top-level RAG configuration."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    synthesizer: SynthesizerConfig = Field(default_factory=SynthesizerConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> RAGConfig:
        """Load config from a YAML file."""
        import yaml

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)
