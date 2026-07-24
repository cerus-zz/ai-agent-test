"""Dataset loaders for RAG evaluation.

Supports downloading Natural Questions and HotpotQA subsets via HuggingFace datasets.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class QAPair:
    """Unified QA pair format across all datasets."""

    id: str
    question: str
    answer: str | list[str]  # single answer or list for multi-answer
    context: str | None = None  # gold context, if available
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentSet:
    """A set of documents/passages for indexing."""

    id: str
    content: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def load_natural_questions(subset: str = "dev", max_samples: int = 500) -> list[QAPair]:
    """Load Natural Questions from HuggingFace datasets.

    Uses the nq_open variant which is freely available.
    Reference: https://huggingface.co/datasets/google-research-datasets/nq_open

    Args:
        subset: 'train', 'dev', or 'test'
        max_samples: Maximum number of samples to load
    """
    from datasets import load_dataset

    logger.info(f"Loading Natural Questions (nq_open/{subset}), max_samples={max_samples}")
    ds = load_dataset("google-research-datasets/nq_open", split=subset, streaming=False)

    pairs = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        pairs.append(
            QAPair(
                id=f"nq_{i}",
                question=row["question"],
                answer=row["answer"],  # nq_open provides list of answers
                metadata={"source": "natural_questions", "subset": subset},
            )
        )
    logger.info(f"Loaded {len(pairs)} NQ samples")
    return pairs


def load_hotpotqa(subset: str = "validation", max_samples: int = 500, difficulty: str | None = None) -> list[QAPair]:
    """Load HotpotQA from HuggingFace datasets.

    Reference: https://huggingface.co/datasets/hotpotqa/hotpot_qa

    Args:
        subset: 'train' or 'validation'
        max_samples: Maximum number of samples to load
        difficulty: Filter by difficulty ('easy', 'medium', 'hard'), or None for all
    """
    from datasets import load_dataset

    logger.info(f"Loading HotpotQA (hotpot_qa/{subset}), max_samples={max_samples}, difficulty={difficulty}")
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split=subset, streaming=False)

    pairs = []
    count = 0
    for row in ds:
        if count >= max_samples:
            break
        if difficulty and row.get("level") != difficulty:
            continue
        pairs.append(
            QAPair(
                id=row.get("_id", f"hp_{count}"),
                question=row["question"],
                answer=row["answer"],
                metadata={
                    "source": "hotpotqa",
                    "subset": subset,
                    "type": row.get("type", ""),
                    "level": row.get("level", ""),
                },
            )
        )
        count += 1
    logger.info(f"Loaded {len(pairs)} HotpotQA samples")
    return pairs


def load_dataset(name: str, **kwargs) -> list[QAPair]:
    """Unified loader: load a dataset by name.

    Args:
        name: 'natural_questions' or 'hotpotqa'
        **kwargs: passed to the specific loader
    """
    loaders = {
        "natural_questions": load_natural_questions,
        "hotpotqa": load_hotpotqa,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    return loaders[name](**kwargs)
