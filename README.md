# My Agent — RAG Experiment Framework

A framework-agnostic RAG experimentation platform. Research and evaluate different RAG strategies (Naive RAG, HyDE, Graph RAG, ...) against public benchmark datasets, without being locked into any single agent framework.

## Design Philosophy

**Core vs. Framework — Clean Separation**

```
  adapters/  ──┐
                │ depends on
  core/      ◄──┘
```

- `core/` contains pure Python logic: retrieval, synthesis, pipeline orchestration. It depends **only** on standard ML libraries (OpenAI SDK, ChromaDB, NetworkX). It knows nothing about agent frameworks.
- `adapters/` are thin wrappers (~50 lines each) that map core functions onto framework-specific primitives (LangGraph nodes, CrewAI tasks, etc.). Swapping frameworks means writing a new adapter, not rewriting logic.
- Tests target `core/`, not frameworks — your RAG logic is what matters.

## Operational Modes

The entry point (`my_agent.main`) supports two primary operational modes:

1. **`rag` Mode**: Interactive dialogue or single query mode. Users can specify any RAG strategy (`naive`, `hyde`, `graphrag`) to process questions.
2. **`eval` Mode**: Run benchmark evaluation experiments across standard question datasets to compute RAGAS & simple metrics.

## Project Structure

```
my_agent/
├── config/
│   ├── schema.py              # Pydantic config models
│   ├── default.yaml           # Base config (API keys, model, DB paths)
│   └── rag_experiments/       # Per-strategy config overrides
│       ├── naive_rag.yaml
│       └── hyde.yaml
│
├── core/                      # Framework-agnostic logic
│   ├── base.py                # Abstract classes & dataclasses
│   ├── retriever.py           # ChromaRetriever + HyDERetriever
│   ├── synthesizer.py         # LLM-based answer generation
│   ├── pipeline.py            # Orchestration: retrieve → synthesize
│   ├── memory_store.py        # DictMemory (swap for Redis)
│   └── graph_builder.py       # NetworkX-based knowledge graph (Graph RAG stub)
│
├── adapters/                  # Thin framework wrappers
│   └── langgraph_adapter.py   # Maps pipeline → LangGraph StateGraph nodes
│
├── data/
│   └── loaders.py             # Decoupled Corpus & QA loaders for EnterpriseRAG-Bench, WixQA, T²-RAGBench
│
├── eval/
│   ├── metrics.py             # RAGAS + simple exact-match/contains metrics
│   └── runner.py              # Batch experiment runner, saves JSON results
│
├── main.py                    # CLI entry point (rag / eval modes)
│
tests/
├── test_retriever.py          # Tests core — no agent framework involved
├── test_pipeline.py
└── test_adapters.py

environment.yml                # conda environment
setup.bat / setup.sh           # One-command setup
```

## What We Have Built

### Retrieval Strategies

| Strategy | File | How It Works |
|----------|------|---------------|
| **Naive RAG** | `core/retriever.py → ChromaRetriever` | Query → embed → cosine similarity search in ChromaDB |
| **HyDE** | `core/retriever.py → HyDERetriever` | Query → LLM generates hypothetical answer → embed hypothetical → vector search |
| **Graph RAG** | `adapters/langgraph_adapter.py` | StateGraph execution combining graph structure & retrieval nodes |

### Datasets

- **EnterpriseRAG-Bench** (`onyx-dot-app/EnterpriseRAG-Bench`) — Real-world company internal knowledge benchmark
- **WixQA** (`Wix/WixQA`) — Domain-specific customer support KB benchmark
- **T²-RAGBench** (`G4KMU/t2-ragbench`) — Financial document benchmark with text and tabular data

Corpus documents (`load_corpus`) and evaluation QA pairs (`load_qa_dataset`) are decoupled and independently disk-cached under `my_agent/data/cache/` so corpus downloading happens once.

### Evaluation

- **RAGAS metrics**: faithfulness, answer relevancy, context precision, context recall
- **Simple metrics**: exact match, contains_answer (no LLM required, fast sanity check)
- Results saved as timestamped JSON under `experiments/results/`

## Quick Start

### 1. Setup

```bash
# Install libmamba solver first (one-time)
conda install -n base conda-libmamba-solver

# Create environment
setup.bat   # Windows
# or
bash setup.sh  # Linux/Mac

# Activate
conda activate my_agent
```

### 2. Configure

Edit `my_agent/config/default.yaml` with your OpenAI-compatible API credentials:

```yaml
llm:
  base_url: "https://api.openai.com/v1"
  api_key: "sk-your-key-here"
  model: "gpt-4o-mini"
```

### 3. Dialogue / Single Query (`rag` mode)

```bash
# Interactive dialogue using Naive RAG strategy (with optional WixQA corpus indexing)
python -m my_agent.main rag --strategy naive --dataset wixqa --interactive

# Single query using HyDE strategy
python -m my_agent.main rag --strategy hyde --query "How do I add a custom domain?"

# Single query using GraphRAG strategy
python -m my_agent.main rag --strategy graphrag --query "Explain our pricing plans"
```

### 4. Run Benchmark Evaluations (`eval` mode)

```bash
# Naive RAG on WixQA
python -m my_agent.main eval --dataset wixqa --experiment naive_rag --max-samples 50

# HyDE on EnterpriseRAG-Bench
python -m my_agent.main eval --dataset enterpriserag_bench --experiment hyde --max-samples 50

# Compare results in experiments/results/
```

### 5. Run Tests

```bash
pytest tests/ -v
```

## Roadmap

- [ ] Graph RAG: integrate `graph_builder.py` with retrieval pipeline
- [ ] More adapters: CrewAI, AutoGen
- [ ] More strategies: Self-RAG, Corrective RAG, Agentic RAG
- [ ] Result visualization & comparison dashboards
