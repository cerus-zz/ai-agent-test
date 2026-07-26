"""Dataset loaders for enterprise RAG benchmarks.

Supports decoupled loading and caching of corpus documents and evaluation QA pairs:
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
HF_CACHE_DIR = DATA_DIR / "hf_cache"


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


# ---------------------------------------------------------------------------
# Disk Caching Helpers
# ---------------------------------------------------------------------------


def _save_corpus_cache(corpus_docs: list[Document], cache_path: Path) -> None:
    """Persist corpus documents to disk cache as JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "id": doc.id,
            "content": doc.content,
            "metadata": doc.metadata,
            "embedding": doc.embedding,
        }
        for doc in corpus_docs
    ]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info(f"Cached {len(corpus_docs)} corpus documents → {cache_path}")


def _load_corpus_cache(cache_path: Path) -> list[Document] | None:
    """Load corpus documents from disk cache if existing."""
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    docs = [
        Document(
            id=d["id"],
            content=d["content"],
            metadata=d.get("metadata", {}),
            embedding=d.get("embedding"),
        )
        for d in data
    ]
    logger.info(f"Loaded {len(docs)} corpus documents from cache ({cache_path})")
    return docs


def _save_qa_cache(qa_pairs: list[QAPair], cache_path: Path) -> None:
    """Persist QA pairs to disk cache as JSON."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(p) for p in qa_pairs]
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    logger.info(f"Cached {len(qa_pairs)} QA pairs → {cache_path}")


def _load_qa_cache(cache_path: Path) -> list[QAPair] | None:
    """Load QA pairs from disk cache if existing."""
    if not cache_path.exists():
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    qa_pairs = [QAPair(**row) for row in data]
    logger.info(f"Loaded {len(qa_pairs)} QA pairs from cache ({cache_path})")
    return qa_pairs


# ---------------------------------------------------------------------------
# Corpus Loaders
# ---------------------------------------------------------------------------


def load_enterpriserag_corpus(force_reload: bool = False) -> list[Document]:
    """Load EnterpriseRAG-Bench corpus documents."""
    cache_path = CACHE_DIR / "corpus_enterpriserag_bench.json"
    if not force_reload:
        cached = _load_corpus_cache(cache_path)
        if cached is not None:
            return cached

    from datasets import load_dataset as hf_load_dataset

    logger.info("Loading EnterpriseRAG-Bench corpus documents...")
    corpus_docs: list[Document] = []
    try:
        ds_corpus = hf_load_dataset(
            "onyx-dot-app/EnterpriseRAG-Bench",
            name="documents",
            split="train",
            streaming=False,
            cache_dir=str(HF_CACHE_DIR),
        )
        for row in ds_corpus:
            doc_id = str(row.get("doc_id") or row.get("id") or "")
            content = row.get("text") or row.get("content") or row.get("body") or ""
            if doc_id and content:
                corpus_docs.append(
                    Document(
                        id=doc_id,
                        content=content,
                        metadata={"source": "enterpriserag_bench", "doc_id": doc_id, "title": row.get("title", "")},
                    )
                )
    except Exception as e:
        logger.warning(f"Could not load EnterpriseRAG-Bench corpus documents dataset: {e}")

    _save_corpus_cache(corpus_docs, cache_path)
    return corpus_docs


def load_wixqa_corpus(force_reload: bool = False) -> list[Document]:
    """Load WixQA corpus documents."""
    cache_path = CACHE_DIR / "corpus_wixqa.json"
    if not force_reload:
        cached = _load_corpus_cache(cache_path)
        if cached is not None:
            return cached

    from datasets import load_dataset as hf_load_dataset

    logger.info("Loading WixQA corpus documents (wix_kb_corpus)...")
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

    _save_corpus_cache(corpus_docs, cache_path)
    return corpus_docs


def load_t2_ragbench_corpus(subset: str = "FinQA", split: str = "train", force_reload: bool = False) -> list[Document]:
    """Load T²-RAGBench corpus documents."""
    cache_path = CACHE_DIR / f"corpus_t2_ragbench_{subset}_{split}.json"
    if not force_reload:
        cached = _load_corpus_cache(cache_path)
        if cached is not None:
            return cached

    from datasets import load_dataset as hf_load_dataset

    logger.info(f"Loading T²-RAGBench corpus documents ({subset}/{split})...")
    ds = hf_load_dataset(
        "G4KMU/t2-ragbench",
        name=subset,
        split=split,
        streaming=False,
        cache_dir=str(HF_CACHE_DIR),
    )

    corpus_docs_dict: dict[str, Document] = {}
    for i, row in enumerate(ds):
        context_id = str(row.get("context_id", f"ctx_{i}"))
        pre_text = row.get("pre_text", "")
        post_text = row.get("post_text", "")
        table = row.get("table", "")
        context_text = row.get("context", "")
        full_doc_content = f"{pre_text}\n{table}\n{post_text}\n{context_text}".strip()

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

    corpus_docs = list(corpus_docs_dict.values())
    _save_corpus_cache(corpus_docs, cache_path)
    return corpus_docs


def load_corpus(name: str, **kwargs) -> list[Document]:
    """Unified corpus loader."""
    loaders = {
        "enterpriserag_bench": load_enterpriserag_corpus,
        "wixqa": load_wixqa_corpus,
        "t2_ragbench": load_t2_ragbench_corpus,
    }
    if name not in loaders:
        raise ValueError(f"Unknown corpus: {name}. Available: {list(loaders.keys())}")
    return loaders[name](**kwargs)


# ---------------------------------------------------------------------------
# QA Loaders
# ---------------------------------------------------------------------------


def load_enterpriserag_qa(
    subset: str = "questions",
    split: str = "test",
    max_samples: int = 50,
    force_reload: bool = False,
) -> list[QAPair]:
    """Load EnterpriseRAG-Bench QA pairs."""
    cache_path = CACHE_DIR / f"qa_enterpriserag_{subset}_{split}_{max_samples}.json"
    if not force_reload:
        cached = _load_qa_cache(cache_path)
        if cached is not None:
            return cached

    from datasets import load_dataset as hf_load_dataset

    logger.info(f"Loading EnterpriseRAG-Bench QA pairs ({subset}/{split}), max_samples={max_samples}")
    ds = hf_load_dataset(
        "onyx-dot-app/EnterpriseRAG-Bench",
        name=subset,
        split=split,
        streaming=False,
        cache_dir=str(HF_CACHE_DIR),
    )

    qa_pairs: list[QAPair] = []
    for i, row in enumerate(ds):
        if i >= max_samples:
            break
        qid = str(row.get("question_id", f"erag_{i}"))
        doc_ids = row.get("expected_doc_ids", [])
        if isinstance(doc_ids, str):
            doc_ids = [doc_ids]
        doc_ids = [str(d) for d in doc_ids]

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

    _save_qa_cache(qa_pairs, cache_path)
    return qa_pairs


def load_wixqa_qa(
    subset: str = "wixqa_expertwritten",
    split: str = "train",
    max_samples: int = 50,
    force_reload: bool = False,
) -> list[QAPair]:
    """Load WixQA QA pairs."""
    cache_path = CACHE_DIR / f"qa_wixqa_{subset}_{split}_{max_samples}.json"
    if not force_reload:
        cached = _load_qa_cache(cache_path)
        if cached is not None:
            return cached

    from datasets import load_dataset as hf_load_dataset

    logger.info(f"Loading WixQA QA pairs ({subset}/{split}), max_samples={max_samples}")
    ds_qa = hf_load_dataset(
        "Wix/WixQA",
        name=subset,
        split=split,
        streaming=False,
        cache_dir=str(HF_CACHE_DIR),
    )
    qa_pairs: list[QAPair] = []

    for i, row in enumerate(ds_qa):
        if i >= max_samples:
            break
        art_ids = row.get("article_ids", [])
        if isinstance(art_ids, str):
            art_ids = [art_ids]
        art_ids_str = [str(x) for x in art_ids]

        qa_pairs.append(
            QAPair(
                id=f"wix_{i}",
                question=row["question"],
                answer=row.get("answer", ""),
                ground_truth_doc_ids=art_ids_str,
                metadata={"source": "wixqa", "subset": subset},
            )
        )

    _save_qa_cache(qa_pairs, cache_path)
    return qa_pairs


def load_t2_ragbench_qa(
    subset: str = "FinQA",
    split: str = "train",
    max_samples: int = 50,
    force_reload: bool = False,
) -> list[QAPair]:
    """Load T²-RAGBench QA pairs."""
    cache_path = CACHE_DIR / f"qa_t2_ragbench_{subset}_{split}_{max_samples}.json"
    if not force_reload:
        cached = _load_qa_cache(cache_path)
        if cached is not None:
            return cached

    from datasets import load_dataset as hf_load_dataset

    logger.info(f"Loading T²-RAGBench QA pairs ({subset}/{split}), max_samples={max_samples}")
    ds = hf_load_dataset(
        "G4KMU/t2-ragbench",
        name=subset,
        split=split,
        streaming=False,
        cache_dir=str(HF_CACHE_DIR),
    )

    qa_pairs: list[QAPair] = []
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

    _save_qa_cache(qa_pairs, cache_path)
    return qa_pairs


def load_qa_dataset(name: str, max_samples: int = 50, force_reload: bool = False, **kwargs) -> list[QAPair]:
    """Unified QA loader: load QA pairs by dataset name."""
    loaders = {
        "enterpriserag_bench": load_enterpriserag_qa,
        "wixqa": load_wixqa_qa,
        "t2_ragbench": load_t2_ragbench_qa,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(loaders.keys())}")
    return loaders[name](max_samples=max_samples, force_reload=force_reload, **kwargs)


def load_dataset_bundle(name: str, max_samples: int = 50, force_reload: bool = False, **kwargs) -> DatasetBundle:
    """Unified loader: loads both QA pairs and corpus documents into a DatasetBundle."""
    qa_pairs = load_qa_dataset(name, max_samples=max_samples, force_reload=force_reload, **kwargs)
    corpus_docs = load_corpus(name, force_reload=force_reload, **kwargs)

    return DatasetBundle(
        name=name,
        qa_pairs=qa_pairs,
        corpus_docs=corpus_docs,
        metadata={"name": name, "max_samples": max_samples},
    )
