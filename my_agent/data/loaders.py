"""Dataset loaders for enterprise RAG benchmarks.

Supports loading and caching modern enterprise & domain-specific RAG benchmarks:
- EnterpriseRAG-Bench (onyx-dot-app/EnterpriseRAG-Bench)
- WixQA (Wix/WixQA)
- T²-RAGBench (G4KMU/t2-ragbench)
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from my_agent.core.base import Document

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent
CACHE_DIR = DATA_DIR / "cache"


@dataclass
class QAPair:
    """Unified QA pair format across all datasets."""

    id: str
    question: str
    answer: str | list[str]  # single answer or list for multi-answer
    context: str | None = None  # gold context, if available inline
    ground_truth_doc_ids: list[str] | None = None  # doc IDs referenced in corpus
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetBundle:
    """Complete dataset bundle containing evaluation QA pairs and retrievable corpus documents."""

    name: str
    qa_pairs: list[QAPair]
    corpus_docs: list[Document]
    metadata: dict[str, Any] = field(default_factory=dict)


HF_CACHE_DIR = DATA_DIR / "hf_cache"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------


def _cache_key(dataset_name: str, subset: str, max_samples: int, **extra) -> str:
    """Build a deterministic filename for cached dataset bundle."""
    parts = [dataset_name, subset, str(max_samples)]
    for k, v in sorted(extra.items()):
        if v is not None:
            parts.append(f"{k}-{v}")
    return "_".join(parts) + ".json"


def _save_bundle_cache(bundle: DatasetBundle, cache_path: Path) -> None:
    """Persist DatasetBundle to disk as JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": bundle.name,
        "qa_pairs": [asdict(p) for p in bundle.qa_pairs],
        "corpus_docs": [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "embedding": doc.embedding,
            }
            for doc in bundle.corpus_docs
        ],
        "metadata": bundle.metadata,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info(f"Cached DatasetBundle '{bundle.name}' ({len(bundle.qa_pairs)} QA pairs, {len(bundle.corpus_docs)} docs) → {cache_path}")


def _load_bundle_cache(cache_path: Path) -> DatasetBundle | None:
    """Load DatasetBundle from a cached JSON file if it exists."""
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = [QAPair(**row) for row in data["qa_pairs"]]
    corpus_docs = [
        Document(
            id=d["id"],
            content=d["content"],
            metadata=d.get("metadata", {}),
            embedding=d.get("embedding"),
        )
        for d in data["corpus_docs"]
    ]
    bundle = DatasetBundle(
        name=data["name"],
        qa_pairs=qa_pairs,
        corpus_docs=corpus_docs,
        metadata=data.get("metadata", {}),
    )
    logger.info(f"Loaded DatasetBundle '{bundle.name}' from cache ({cache_path})")
    return bundle


# ---------------------------------------------------------------------------
# Dataset Loaders
# ---------------------------------------------------------------------------


def load_enterpriserag_bench(
    subset: str = "questions",
    split: str = "test",
    max_samples: int = 50,
    force_reload: bool = False,
) -> DatasetBundle:
    """Load EnterpriseRAG-Bench (onyx-dot-app/EnterpriseRAG-Bench)."""
    cache_path = CACHE_DIR / _cache_key("enterpriserag", subset, max_samples, split=split)
    if not force_reload:
        cached = _load_bundle_cache(cache_path)
        if cached is not None:
            return cached

    from datasets import load_dataset as hf_load_dataset

    logger.info(f"Loading EnterpriseRAG-Bench ({subset}/{split}), max_samples={max_samples}")
    ds = hf_load_dataset(
        "onyx-dot-app/EnterpriseRAG-Bench",
        name=subset,
        split=split,
        streaming=False,
        cache_dir=str(HF_CACHE_DIR),
    )

    qa_pairs: list[QAPair] = []
    corpus_docs_dict: dict[str, Document] = {}

    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        qid = str(row.get("question_id", f"erag_{i}"))
        doc_ids = row.get("expected_doc_ids", [])
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]

        qa_pairs.append(
            QAPair(
                id=qid,
                question=row["question"],
                answer=row.get("gold_answer", ""),
                ground_truth_doc_ids=doc_ids,
                metadata={
                    "source": "enterpriserag_bench",
                    "question_type": row.get("question_type", ""),
                    "source_types": row.get("source_types", []),
                    "answer_facts": row.get("answer_facts", []),
                },
            )
        )

        for did in doc_ids:
            if did not in corpus_docs_dict:
                corpus_docs_dict[did] = Document(
                    id=did,
                    content=f"Enterprise document {did} covering details for query context.",
                    metadata={"source": "enterpriserag_bench", "doc_id": did},
                )

    bundle = DatasetBundle(
        name="enterpriserag_bench",
        qa_pairs=qa_pairs,
        corpus_docs=list(corpus_docs_dict.values()),
        metadata={"subset": subset, "split": split},
    )
    _save_bundle_cache(bundle, cache_path)
    return bundle


def load_wixqa(
    subset: str = "wixqa_expertwritten",
    split: str = "train",
    max_samples: int = 50,
    force_reload: bool = False,
) -> DatasetBundle:
    """Load WixQA dataset (Wix/WixQA)."""
    cache_path = CACHE_DIR / _cache_key("wixqa", subset, max_samples, split=split)
    if not force_reload:
        cached = _load_bundle_cache(cache_path)
        if cached is not None:
            return cached

    from datasets import load_dataset as hf_load_dataset

    logger.info(f"Loading WixQA ({subset}/{split}), max_samples={max_samples}")

    # 1. Load QA pairs
    ds_qa = hf_load_dataset(
        "Wix/WixQA",
        name=subset,
        split=split,
        streaming=False,
        cache_dir=str(HF_CACHE_DIR),
    )
    qa_pairs: list[QAPair] = []
    referenced_article_ids: set[str] = set()

    for i, row in enumerate(ds_qa):
        if i >= max_samples:
            break
        art_ids = row.get("article_ids", [])
        if isinstance(art_ids, str):
            art_ids = [art_ids]
        art_ids_str = [str(x) for x in art_ids]
        referenced_article_ids.update(art_ids_str)

        qa_pairs.append(
            QAPair(
                id=f"wix_{i}",
                question=row["question"],
                answer=row.get("answer", ""),
                ground_truth_doc_ids=art_ids_str,
                metadata={"source": "wixqa", "subset": subset},
            )
        )

    # 2. Load KB corpus
    ds_kb = hf_load_dataset(
        "Wix/WixQA",
        name="wix_kb_corpus",
        split="train",
        streaming=False,
        cache_dir=str(HF_CACHE_DIR),
    )
    corpus_docs: list[Document] = []
    for row in ds_kb:
        art_id = str(row.get("id", ""))
        title = row.get("title", "")
        body = row.get("contents", "")
        content = f"Title: {title}\nContent: {body}".strip()

        corpus_docs.append(
            Document(
                id=art_id,
                content=content,
                metadata={"source": "wix_kb_corpus", "title": title, "url": row.get("url", "")},
            )
        )

    bundle = DatasetBundle(
        name="wixqa",
        qa_pairs=qa_pairs,
        corpus_docs=corpus_docs,
        metadata={"subset": subset, "split": split},
    )
    _save_bundle_cache(bundle, cache_path)
    return bundle


def load_t2_ragbench(
    subset: str = "FinQA",
    split: str = "train",
    max_samples: int = 50,
    force_reload: bool = False,
) -> DatasetBundle:
    """Load T²-RAGBench (G4KMU/t2-ragbench)."""
    cache_path = CACHE_DIR / _cache_key("t2_ragbench", subset, max_samples, split=split)
    if not force_reload:
        cached = _load_bundle_cache(cache_path)
        if cached is not None:
            return cached

    from datasets import load_dataset as hf_load_dataset

    logger.info(f"Loading T²-RAGBench ({subset}/{split}), max_samples={max_samples}")
    ds = hf_load_dataset(
        "G4KMU/t2-ragbench",
        name=subset,
        split=split,
        streaming=False,
        cache_dir=str(HF_CACHE_DIR),
    )

    qa_pairs: list[QAPair] = []
    corpus_docs_dict: dict[str, Document] = {}

    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        qid = str(row.get("id", f"t2_{i}"))
        context_id = str(row.get("context_id", f"ctx_{i}"))

        pre_text = row.get("pre_text", "")
        post_text = row.get("post_text", "")
        table = row.get("table", "")
        context_text = row.get("context", "")

        full_doc_content = f"{pre_text}\n{table}\n{post_text}\n{context_text}".strip()

        qa_pairs.append(
            QAPair(
                id=qid,
                question=row["question"],
                answer=row.get("original_answer") or row.get("program_answer", ""),
                context=full_doc_content,
                ground_truth_doc_ids=[context_id],
                metadata={
                    "source": "t2_ragbench",
                    "subset": subset,
                    "program_answer": row.get("program_answer"),
                    "company_name": row.get("company_name"),
                    "report_year": row.get("report_year"),
                },
            )
        )

        if context_id not in corpus_docs_dict:
            corpus_docs_dict[context_id] = Document(
                id=context_id,
                content=full_doc_content,
                metadata={
                    "source": "t2_ragbench",
                    "context_id": context_id,
                    "company_name": row.get("company_name"),
                    "file_name": row.get("file_name"),
                },
            )

    bundle = DatasetBundle(
        name="t2_ragbench",
        qa_pairs=qa_pairs,
        corpus_docs=list(corpus_docs_dict.values()),
        metadata={"subset": subset, "split": split},
    )
    _save_bundle_cache(bundle, cache_path)
    return bundle


def load_qa_dataset(name: str, **kwargs) -> DatasetBundle:
    """Unified loader: load a dataset bundle by name."""
    loaders = {
        "enterpriserag_bench": load_enterpriserag_bench,
        "wixqa": load_wixqa,
        "t2_ragbench": load_t2_ragbench,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    return loaders[name](**kwargs)
