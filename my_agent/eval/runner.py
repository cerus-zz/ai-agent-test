"""Experiment runner: evaluates RAG strategies against datasets."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

from my_agent.config.schema import RAGConfig
from my_agent.core.base import Document
from my_agent.core.pipeline import RAGPipeline
from my_agent.data.loaders import QAPair, load_qa_dataset
from my_agent.eval.metrics import EvalResult, compute_ragas_metrics, compute_simple_metrics

logger = logging.getLogger(__name__)


def run_experiment(
    config: RAGConfig,
    dataset_name: str,
    experiment_name: str,
    max_samples: int = 100,
    dataset_kwargs: dict | None = None,
    results_dir: str = "experiments/results",
    force_reload: bool = False,
) -> EvalResult:
    """Run a full evaluation experiment.

    Flow:
    1. Load dataset (QA pairs + context documents)
    2. Index documents into the retriever
    3. For each QA pair: query → retrieve → synthesize
    4. Compute metrics (simple + RAGAS)
    5. Save results to disk

    Args:
        config: RAG configuration
        dataset_name: 'natural_questions' or 'hotpotqa'
        experiment_name: Name for this experiment run
        max_samples: Maximum QA pairs to evaluate
        dataset_kwargs: Extra args for the dataset loader
        results_dir: Directory to save results
    """
    logger.info(f"=== Experiment: {experiment_name} ===")
    logger.info(f"Dataset: {dataset_name}, Strategy: {config.retriever.strategy}")

    dataset_kwargs = dataset_kwargs or {}

    # 1. Load dataset bundle
    bundle = load_qa_dataset(dataset_name, max_samples=max_samples, force_reload=force_reload, **dataset_kwargs)
    qa_pairs = bundle.qa_pairs
    if not qa_pairs:
        logger.error(f"No samples loaded from {dataset_name}")
        return EvalResult(experiment_name=experiment_name, dataset_name=dataset_name, num_samples=0)

    # 2. Index corpus documents into ChromaDB
    pipeline = RAGPipeline(config)
    pipeline.index_documents(bundle.corpus_docs)
    logger.info(f"Indexed {len(bundle.corpus_docs)} corpus documents into ChromaDB")

    # 3. Run queries
    questions = []
    answers = []
    contexts_list = []
    ground_truths = []

    start_time = time.time()
    for i, qa in enumerate(qa_pairs):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i + 1}/{len(qa_pairs)}")

        result = pipeline.query(qa.question)
        questions.append(qa.question)
        answers.append(result.answer)
        contexts_list.append([doc.content for doc in result.retrieved_docs])
        ground_truths.append(qa.answer)

    elapsed = time.time() - start_time
    logger.info(f"Completed {len(qa_pairs)} queries in {elapsed:.1f}s ({elapsed / max(len(qa_pairs), 1):.2f}s/query)")

    # 4. Compute metrics
    simple = compute_simple_metrics(questions, answers, ground_truths)

    ragas = compute_ragas_metrics(
        questions=questions,
        answers=answers,
        contexts=contexts_list,
        ground_truths=ground_truths,
        llm_base_url=config.llm.base_url,
        llm_api_key=config.llm.api_key,
        llm_model=config.llm.model,
    )

    metrics = {**simple, **ragas}
    metrics["avg_query_time_s"] = elapsed / max(len(qa_pairs), 1)

    # 5. Save results
    eval_result = EvalResult(
        experiment_name=experiment_name,
        dataset_name=dataset_name,
        num_samples=len(qa_pairs),
        metrics=metrics,
        per_sample=[
            {"question": q, "answer": a, "ground_truth": gt}
            for q, a, gt in zip(questions, answers, ground_truths)
        ],
    )
    _save_results(eval_result, results_dir)

    # Print summary
    logger.info("--- Results ---")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    return eval_result


def run_experiment_set(
    base_config: RAGConfig,
    dataset_name: str,
    experiments: list[tuple[str, RAGConfig]],
    max_samples: int = 100,
    results_dir: str = "experiments/results",
) -> list[EvalResult]:
    """Run multiple experiments (e.g., naive vs hyde) on the same dataset."""
    results = []
    for name, cfg in experiments:
        result = run_experiment(
            config=cfg,
            dataset_name=dataset_name,
            experiment_name=name,
            max_samples=max_samples,
            results_dir=results_dir,
        )
        results.append(result)
    return results


def _index_dataset_documents(pipeline: RAGPipeline, qa_pairs: list[QAPair]) -> None:
    """Index QA pair questions as retrievable documents.

    In a real scenario, you'd index the actual corpus passages.
    For quick experiments, we index the questions + answers.
    """
    docs = []
    for qa in qa_pairs:
        answer_text = qa.answer if isinstance(qa.answer, str) else " ".join(qa.answer)
        docs.append(
            Document(
                id=qa.id,
                content=f"Q: {qa.question}\nA: {answer_text}",
                metadata={"source": qa.metadata.get("source", ""), "qa_id": qa.id},
            )
        )
    pipeline.index_documents(docs)
    logger.info(f"Indexed {len(docs)} documents")


def _save_results(result: EvalResult, results_dir: str) -> None:
    """Save experiment results to JSON."""
    output_dir = Path(results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{result.experiment_name}_{result.dataset_name}_{timestamp}.json"
    filepath = output_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": result.experiment_name,
                "dataset_name": result.dataset_name,
                "num_samples": result.num_samples,
                "metrics": result.metrics,
                "per_sample": result.per_sample,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    logger.info(f"Results saved to {filepath}")
