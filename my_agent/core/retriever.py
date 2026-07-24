"""Retriever implementations: Naive (vector search) and HyDE."""

from __future__ import annotations

import uuid
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from my_agent.config.schema import ChromaConfig, EmbeddingConfig, RetrieverConfig
from my_agent.core.base import BaseRetriever, Document, RetrievedDocument


class EmbeddingProvider:
    """Unified embedding interface supporting sentence-transformers and OpenAI."""

    def __init__(self, config: EmbeddingConfig):
        self.config = config
        self._model = None

    @property
    def model(self):
        if self._model is None:
            if self.config.provider == "sentence-transformers":
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.config.model_name)
            else:
                self._model = self  # use self.embed for OpenAI
        return self._model

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, return list of embedding vectors."""
        if self.config.provider == "sentence-transformers":
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()

        # OpenAI-compatible embedding
        from openai import OpenAI

        client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        resp = client.embeddings.create(model=self.config.model_name, input=texts)
        return [d.embedding for d in resp.data]

    def embed_query(self, text: str) -> list[float]:
        return self.embed([text])[0]


class ChromaRetriever(BaseRetriever):
    """Naive retriever using ChromaDB for vector similarity search."""

    def __init__(self, chroma_config: ChromaConfig, embedding_config: EmbeddingConfig):
        self.chroma_config = chroma_config
        self.embedder = EmbeddingProvider(embedding_config)
        self._client = chromadb.PersistentClient(
            path=chroma_config.persist_directory,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = None

    @property
    def collection(self):
        if self._collection is None:
            self._collection = self._client.get_or_create_collection(
                name=self.chroma_config.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def index(self, documents: list[Document]) -> None:
        if not documents:
            return

        ids = [doc.id or str(uuid.uuid4()) for doc in documents]
        contents = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        # Embed documents if they don't have pre-computed embeddings
        if documents[0].embedding is None:
            embeddings = self.embedder.embed(contents)
        else:
            embeddings = [doc.embedding for doc in documents]

        self.collection.add(ids=ids, documents=contents, embeddings=embeddings, metadatas=metadatas)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)

        docs = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                docs.append(
                    RetrievedDocument(
                        id=results["ids"][0][i],
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        score=1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                    )
                )
        return docs

    def clear(self) -> None:
        self._client.delete_collection(name=self.chroma_config.collection_name)
        self._collection = None


class HyDERetriever(BaseRetriever):
    """HyDE retriever: generates hypothetical document, then retrieves by its embedding.

    Flow: query → LLM generates hypothetical answer/doc → embed that doc → vector search.
    """

    def __init__(
        self,
        chroma_config: ChromaConfig,
        embedding_config: EmbeddingConfig,
        llm_base_url: str = "https://api.openai.com/v1",
        llm_api_key: str = "sk-xxx",
        llm_model: str = "gpt-4o-mini",
    ):
        self._chroma_retriever = ChromaRetriever(chroma_config, embedding_config)
        self.embedder = self._chroma_retriever.embedder
        self._llm_base_url = llm_base_url
        self._llm_api_key = llm_api_key
        self._llm_model = llm_model

    def _generate_hypothetical_doc(self, query: str) -> str:
        """Use LLM to generate a hypothetical document that answers the query."""
        from openai import OpenAI

        client = OpenAI(base_url=self._llm_base_url, api_key=self._llm_api_key)
        prompt = (
            "You are a knowledgeable assistant. Generate a short, factual passage "
            "that answers the following question. Write only the passage, no preamble.\n\n"
            f"Question: {query}\n\nPassage:"
        )
        resp = client.chat.completions.create(
            model=self._llm_model, messages=[{"role": "user", "content": prompt}], temperature=0.3, max_tokens=512
        )
        return resp.choices[0].message.content.strip()

    def index(self, documents: list[Document]) -> None:
        self._chroma_retriever.index(documents)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedDocument]:
        hypothetical = self._generate_hypothetical_doc(query)
        # Embed the hypothetical document and use it as the query vector
        hypo_embedding = self.embedder.embed_query(hypothetical)
        results = self._chroma_retriever.collection.query(
            query_embeddings=[hypo_embedding], n_results=top_k
        )

        docs = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                docs.append(
                    RetrievedDocument(
                        id=results["ids"][0][i],
                        content=results["documents"][0][i] if results["documents"] else "",
                        metadata=results["metadatas"][0][i] if results["metadatas"] else {},
                        score=1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                    )
                )
        return docs

    def clear(self) -> None:
        self._chroma_retriever.clear()


def create_retriever(config: RetrieverConfig, chroma_config: ChromaConfig, embedding_config: EmbeddingConfig, llm_base_url: str = "", llm_api_key: str = "", llm_model: str = "") -> BaseRetriever:
    """Factory function to create the appropriate retriever based on config."""
    if config.strategy == "naive":
        return ChromaRetriever(chroma_config, embedding_config)
    elif config.strategy == "hyde":
        return HyDERetriever(chroma_config, embedding_config, llm_base_url or "https://api.openai.com/v1", llm_api_key or "sk-xxx", llm_model or "gpt-4o-mini")
    else:
        raise ValueError(f"Unknown retriever strategy: {config.strategy}")
