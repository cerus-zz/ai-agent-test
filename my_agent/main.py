"""Entry point: picks a config + strategy and runs in 'rag' or 'eval' mode."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from my_agent.config.schema import RAGConfig
from my_agent.core.pipeline import RAGPipeline
from my_agent.eval.runner import run_experiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("my_agent")


def parse_args():
    parser = argparse.ArgumentParser(description="My Agent - RAG runner & evaluation tool")
    sub = parser.add_subparsers(dest="command")

    # eval command
    eval_parser = sub.add_parser("eval", help="Run a benchmark evaluation experiment")
    eval_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    eval_parser.add_argument("--experiment", "--strategy", dest="experiment", type=str, default=None, help="Strategy/Experiment (naive, hyde, graphrag)")
    eval_parser.add_argument("--dataset", type=str, default="wixqa", choices=["enterpriserag_bench", "wixqa", "t2_ragbench"], help="Dataset name")
    eval_parser.add_argument("--max-samples", type=int, default=50, help="Max QA pairs to evaluate")
    eval_parser.add_argument("--results-dir", type=str, default="experiments/results", help="Results output directory")
    eval_parser.add_argument("--force-reload", action="store_true", default=False, help="Force re-download dataset & corpus")

    # rag command
    rag_parser = sub.add_parser("rag", help="Run in dialogue or single-query RAG mode")
    rag_parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML")
    rag_parser.add_argument("--strategy", type=str, default="naive", help="RAG strategy to use: naive, hyde, graphrag")
    rag_parser.add_argument("--query", type=str, default=None, help="Single query to run. If omitted, enters interactive mode.")
    rag_parser.add_argument("--interactive", action="store_true", default=False, help="Force interactive dialogue mode")
    rag_parser.add_argument("--dataset", type=str, default=None, choices=["enterpriserag_bench", "wixqa", "t2_ragbench"], help="Corpus dataset to load and index before querying")
    rag_parser.add_argument("--force-reload", action="store_true", default=False, help="Force re-download corpus cache")

    return parser.parse_args()


def load_config(config_path: str, strategy: str | None = None) -> RAGConfig:
    """Load base config, optionally overlaid with strategy/experiment config."""
    path_obj = Path(config_path)
    if not path_obj.exists():
        pkg_root = Path(__file__).resolve().parent
        if (pkg_root / config_path).exists():
            path_obj = pkg_root / config_path
        elif (pkg_root / "config" / Path(config_path).name).exists():
            path_obj = pkg_root / "config" / Path(config_path).name
        elif (pkg_root.parent / config_path).exists():
            path_obj = pkg_root.parent / config_path

    base = RAGConfig.from_yaml(path_obj)

    if strategy:
        # map strategy name to config file
        exp_name = strategy
        if exp_name in ["naive", "naive_rag"]:
            exp_name = "naive_rag"

        exp_path = Path(f"config/rag_experiments/{exp_name}.yaml")
        if not exp_path.exists():
            pkg_root = Path(__file__).resolve().parent
            if (pkg_root / exp_path).exists():
                exp_path = pkg_root / exp_path
            elif (pkg_root.parent / exp_path).exists():
                exp_path = pkg_root.parent / exp_path

        if exp_path.exists():
            exp_config = RAGConfig.from_yaml(exp_path)
            base.retriever.strategy = exp_config.retriever.strategy
            base.retriever.top_k = exp_config.retriever.top_k
            base.graph = exp_config.graph
        else:
            base.retriever.strategy = strategy

    return base


def run_rag_mode(config: RAGConfig, strategy: str, query: str | None = None, dataset: str | None = None, force_reload: bool = False, interactive: bool = False):
    """Execute RAG pipeline in single-query or interactive mode."""
    pipeline = RAGPipeline(config)

    # Optional corpus indexing step
    if dataset:
        from my_agent.data.loaders import load_corpus

        corpus_docs = load_corpus(dataset, force_reload=force_reload)
        if corpus_docs:
            pipeline.index_documents(corpus_docs)
            logger.info(f"Indexed {len(corpus_docs)} documents from corpus '{dataset}'")

    def execute_query(q: str):
        if strategy == "graphrag":
            try:
                from my_agent.adapters.langgraph_adapter import invoke_rag_graph

                res = invoke_rag_graph(pipeline, q)
                return res["answer"], res.get("retrieved_docs", [])
            except Exception as e:
                logger.warning(f"GraphRAG adapter invocation failed ({e}), falling back to core pipeline")
                res = pipeline.query(q)
                return res.answer, res.retrieved_docs
        else:
            res = pipeline.query(q)
            return res.answer, res.retrieved_docs

    if query and not interactive:
        answer, docs = execute_query(query)
        print(f"\n--- RAG Output (Strategy: {strategy}) ---")
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print(f"Retrieved Contexts ({len(docs)}):")
        for idx, doc in enumerate(docs, 1):
            content_snippet = doc.content[:150].replace('\n', ' ')
            print(f"  [{idx}] {content_snippet}...")
    else:
        print(f"\n=== RAG Interactive Mode (Strategy: {strategy}) ===")
        print("Type 'exit' or 'quit' to end session.\n")
        while True:
            try:
                user_input = input("User > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["exit", "quit"]:
                    print("Exiting RAG session.")
                    break

                answer, docs = execute_query(user_input)
                print(f"\nAgent > {answer}")
                print(f"  (Retrieved {len(docs)} context passages)\n")
            except (KeyboardInterrupt, EOFError):
                print("\nExiting RAG session.")
                break


def main():
    args = parse_args()
    if not args.command:
        logger.error("No command specified. Available commands: 'rag' or 'eval'")
        sys.exit(1)

    if args.command == "eval":
        config = load_config(
            getattr(args, "config", "config/default.yaml"),
            getattr(args, "experiment", None),
        )
        logger.info(f"Loaded eval config, strategy={config.retriever.strategy}")
        run_experiment(
            config=config,
            dataset_name=args.dataset,
            experiment_name=args.experiment or config.retriever.strategy,
            max_samples=args.max_samples,
            results_dir=args.results_dir,
            force_reload=args.force_reload,
        )

    elif args.command == "rag":
        config = load_config(
            getattr(args, "config", "config/default.yaml"),
            getattr(args, "strategy", "naive"),
        )
        logger.info(f"Loaded RAG config, strategy={config.retriever.strategy}")
        run_rag_mode(
            config=config,
            strategy=args.strategy,
            query=args.query,
            dataset=args.dataset,
            force_reload=args.force_reload,
            interactive=args.interactive,
        )


if __name__ == "__main__":
    main()
