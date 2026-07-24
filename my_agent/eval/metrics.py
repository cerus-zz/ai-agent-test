"""RAG evaluation metrics using RAGAS and custom metrics."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Aggregated evaluation metrics for a single experiment run."""

    experiment_name: str
    dataset_name: str
    num_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    per_sample: list[dict[str, Any]] = field(default_factory=list)


def compute_ragas_metrics(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str | list[str]],
    llm_base_url: str = "https://api.openai.com/v1",
    llm_api_key: str = "sk-xxx",
    llm_model: str = "gpt-4o-mini",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, float]:
    """Compute RAGAS metrics for the given RAG outputs.

    Metrics:
    - faithfulness: How factually accurate the answer is relative to context
    - answer_relevancy: How relevant the answer is to the question
    - context_precision: How precise the retrieved context is
    - context_recall: How much of the relevant context was retrieved

    Args:
        questions: List of queries
        answers: List of generated answers
        contexts: List of retrieved context passages per query
        ground_truths: List of ground truth answers
        llm_*: LLM config for RAGAS evaluation (uses OpenAI-compatible API)
    """
    try:
        from ragas import evaluate
        from ragas.metrics import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )
        from ragas.llms import LangchainLLMWrapper
        from langchain_openai import ChatOpenAI
    except ImportError as e:
        logger.warning(f"RAGAS dependencies not fully installed: {e}")
        return {}

    # Prepare dataset in RAGAS format
    eval_llm = LangchainLLMWrapper(ChatOpenAI(
        model=llm_model,
        openai_api_key=llm_api_key,
        openai_api_base=llm_base_url,
        temperature=0.0,
    ))

    dataset = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    }

    try:
        result = evaluate(
            dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall(),
            ],
            llm=eval_llm,
        )
        return {k: float(v) for k, v in result.items()}
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {}


def compute_simple_metrics(
    questions: list[str],
    answers: list[str],
    ground_truths: list[str | list[str]],
) -> dict[str, float]:
    """Simple string-matching metrics (no LLM required).

    - exact_match: Proportion of answers that exactly match ground truth
    - contains_answer: Proportion of answers that contain the ground truth
    - avg_answer_length: Average character length of generated answers
    """
    exact_matches = 0
    contains = 0
    total_length = 0

    for ans, gt in zip(answers, ground_truths):
        if isinstance(gt, list):
            gt_strs = [str(g).strip().lower() for g in gt]
            ans_lower = str(ans).strip().lower()
            if ans_lower in gt_strs or any(g in ans_lower for g in gt_strs):
                exact_matches += 1
                contains += 1
            elif any(g in ans_lower for g in gt_strs):
                contains += 1
        else:
            ans_lower = str(ans).strip().lower()
            gt_lower = str(gt).strip().lower()
            if ans_lower == gt_lower:
                exact_matches += 1
                contains += 1
            elif gt_lower in ans_lower:
                contains += 1

        total_length += len(str(ans))

    n = len(answers) if answers else 1
    return {
        "exact_match": exact_matches / n,
        "contains_answer": contains / n,
        "avg_answer_length": total_length / n,
    }
