"""Synthesizer (answer generator) using LLM with prompt chaining."""

from __future__ import annotations

from openai import OpenAI

from my_agent.config.schema import LLMConfig
from my_agent.core.base import BaseSynthesizer, RetrievedDocument

# Built-in prompt templates per strategy
NAIVE_PROMPT = """You are a helpful research assistant. Answer the question based ONLY on the provided context.
If the context does not contain enough information to answer, say "I cannot answer based on the provided context."

Context:
{context}

Question: {query}

Answer:"""

HYDE_PROMPT = """You are a helpful research assistant. Answer the question based ONLY on the provided context.
Note: the context was retrieved using an enhanced search strategy, so it should be relevant.

Context:
{context}

Question: {query}

Answer:"""


class LLMSynthesizer(BaseSynthesizer):
    """LLM-based answer synthesizer using OpenAI-compatible API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(base_url=self.config.base_url, api_key=self.config.api_key)
        return self._client

    def _build_context(self, documents: list[RetrievedDocument]) -> str:
        """Build a context string from retrieved documents."""
        parts = []
        for i, doc in enumerate(documents, 1):
            parts.append(f"[Document {i}] (score: {doc.score:.4f})\n{doc.content}")
        return "\n\n".join(parts)

    def synthesize(self, query: str, documents: list[RetrievedDocument], prompt_template: str | None = None) -> str:
        template = prompt_template or NAIVE_PROMPT
        context = self._build_context(documents)

        prompt = template.format(context=context, query=query)

        resp = self.client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content.strip()
