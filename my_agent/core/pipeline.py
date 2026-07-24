"""Pipeline orchestration: wires retriever + synthesizer together."""

from __future__ import annotations

from dataclasses import dataclass

from my_agent.config.schema import RAGConfig
from my_agent.core.base import BaseRetriever, BaseSynthesizer, Document, QueryResult
from my_agent.core.retriever import create_retriever
from my_agent.core.synthesizer import LLMSynthesizer


@dataclass
class PipelineContext:
    """Context passed through the pipeline, available to adapters for inspection."""

    query: str
    retrieved_docs: list = None
    answer: str = ""

    def __post_init__(self):
        if self.retrieved_docs is None:
            self.retrieved_docs = []


class RAGPipeline:
    """Orchestrates the full RAG flow: indexing and querying.

    Framework-agnostic. Adapters call .query() and map results to their own nodes/tasks.
    """

    def __init__(self, config: RAGConfig):
        self.config = config
        self.retriever: BaseRetriever = create_retriever(
            config=config.retriever,
            chroma_config=config.chroma,
            embedding_config=config.embedding,
            llm_base_url=config.llm.base_url,
            llm_api_key=config.llm.api_key,
            llm_model=config.llm.model,
        )
        self.synthesizer: BaseSynthesizer = LLMSynthesizer(config.llm)

    def index_documents(self, documents: list[Document]) -> None:
        """Index a batch of documents into the retriever."""
        self.retriever.index(documents)

    def index_texts(self, texts: list[str], metadatas: list[dict] | None = None) -> None:
        """Convenience: index raw texts as documents."""
        docs = []
        for i, text in enumerate(texts):
            meta = metadatas[i] if metadatas else {}
            docs.append(Document(id=str(i), content=text, metadata=meta))
        self.index_documents(docs)

    def query(self, question: str) -> QueryResult:
        """Run the full RAG pipeline: retrieve → synthesize → return result.

        This is the single entry point that adapters will call.
        """
        retrieved = self.retriever.retrieve(question, top_k=self.config.retriever.top_k)

        prompt_template = self.config.synthesizer.prompt_template
        answer = self.synthesizer.synthesize(question, retrieved, prompt_template=prompt_template)

        return QueryResult(
            query=question,
            answer=answer,
            retrieved_docs=retrieved,
            metadata={
                "strategy": self.config.retriever.strategy,
                "top_k": self.config.retriever.top_k,
                "model": self.config.llm.model,
            },
        )

    def query_with_context(self, question: str) -> tuple[QueryResult, PipelineContext]:
        """Query and return both the result and the intermediate context.

        Useful for adapters that want to inspect pipeline internals (e.g., LangGraph state).
        """
        ctx = PipelineContext(query=question)
        retrieved = self.retriever.retrieve(question, top_k=self.config.retriever.top_k)
        ctx.retrieved_docs = retrieved

        prompt_template = self.config.synthesizer.prompt_template
        answer = self.synthesizer.synthesize(question, retrieved, prompt_template=prompt_template)
        ctx.answer = answer

        result = QueryResult(
            query=question,
            answer=answer,
            retrieved_docs=retrieved,
            metadata={
                "strategy": self.config.retriever.strategy,
                "top_k": self.config.retriever.top_k,
                "model": self.config.llm.model,
            },
        )
        return result, ctx
