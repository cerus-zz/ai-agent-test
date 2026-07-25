"""Entry point: picks a config + adapter and runs."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from my_agent.config.schema import RAGConfig
from my_agent.core.pipeline import RAGPipeline
from my_agent.eval.runner import run_experiment, run_experiment_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("my_agent")


def parse_args():
    parser = argparse.ArgumentParser(description="My Agent - RAG experiment runner")
    sub = parser.add_subparsers(dest="command")

    # eval command
    eval_parser = sub.add_parser("eval", help="Run a RAG evaluation experiment")
    eval_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    eval_parser.add_argument("--experiment", type=str, default=None, help="Experiment config name (naive_rag, hyde)")
    eval_parser.add_argument("--dataset", type=str, default="wixqa", choices=["enterpriserag_bench", "wixqa", "t2_ragbench"], help="Dataset: enterpriserag_bench, wixqa, t2_ragbench")
    eval_parser.add_argument("--max-samples", type=int, default=50, help="Max QA pairs to evaluate")
    eval_parser.add_argument("--results-dir", type=str, default="experiments/results", help="Results output directory")
    eval_parser.add_argument("--force-reload", action="store_true", default=False, help="Force re-download dataset, bypassing local cache")

    # graph command
    graph_parser = sub.add_parser("graph", help="Run with LangGraph adapter")
    graph_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    graph_parser.add_argument("--query", type=str, required=True, help="Query to run")
    graph_parser.add_argument("--dataset", type=str, default=None, choices=["enterpriserag_bench", "wixqa", "t2_ragbench"], help="Dataset to index before querying")
    graph_parser.add_argument("--max-samples", type=int, default=50, help="Max samples to index if dataset is provided")

    return parser.parse_args()


def load_config(config_path: str, experiment: str | None = None) -> RAGConfig:
    """Load base config, optionally overlaid with experiment config."""
    base = RAGConfig.from_yaml(config_path)

    if experiment:
        exp_path = Path(f"config/rag_experiments/{experiment}.yaml")
        if exp_path.exists():
            exp_config = RAGConfig.from_yaml(exp_path)
            # Merge: experiment config overrides base for non-default fields
            if exp_config.retriever.strategy != "naive" or experiment == "hyde":
                base.retriever.strategy = exp_config.retriever.strategy
            base.retriever.top_k = exp_config.retriever.top_k
            base.graph = exp_config.graph

    return base


def main():
    args = parse_args()
    if not args.command:
        logger.error("No command specified. Use: eval or graph")
        sys.exit(1)

    config = load_config(
        getattr(args, "config", "config/default.yaml"),
        getattr(args, "experiment", None),
    )
    logger.info(f"Loaded config, strategy={config.retriever.strategy}")

    if args.command == "eval":
        run_experiment(
            config=config,
            dataset_name=args.dataset,
            experiment_name=args.experiment or config.retriever.strategy,
            max_samples=args.max_samples,
            results_dir=args.results_dir,
            force_reload=args.force_reload,
        )

    elif args.command == "graph":
        pipeline = RAGPipeline(config)
        if getattr(args, "dataset", None):
            from my_agent.data.loaders import load_qa_dataset

            bundle = load_qa_dataset(args.dataset, max_samples=args.max_samples)
            pipeline.index_documents(bundle.corpus_docs)
            logger.info(f"Indexed {len(bundle.corpus_docs)} corpus documents from dataset '{args.dataset}'")

        from my_agent.adapters.langgraph_adapter import invoke_rag_graph

        result = invoke_rag_graph(pipeline, args.query)
        logger.info(f"Query: {args.query}")
        logger.info(f"Answer: {result['answer']}")
        logger.info(f"Retrieved {len(result['retrieved_docs'])} documents")


if __name__ == "__main__":
    main()
