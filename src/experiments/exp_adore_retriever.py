import json
from pathlib import Path
from typing import Dict, List, Any

from dexter.config.constants import Split
from dexter.data.loaders.RetrieverDataset import RetrieverDataset

from src.experiments.AdoreRetriever import AdoreRetriever
from src.retrieval.contriever_retrieval import run_contriever
from src.generative_model_setup import generate_answer


def _build_corpus_index(corpus) -> Dict[str, Any]:
    """doc_id -> Evidence"""
    return {doc.id(): doc for doc in corpus}


def _evidence_to_dict(ev) -> Dict[str, str]:
    return {
        "doc_id": ev.id(),
        "title": ev.title() or "",
        "text": ev.text() or "",
    }



def run_adore_topk(
    k: int,
    out_path: str,
    model_path: str,
    passage_memmap_path: str,
) -> None:
    """
    Use trained ADORE retriever to get top-k contexts per query
    and save them (no LLM call).
    """
    print(f"[adore_topk] Running ADORE with k={k}")

    # Load dataset via Dexter
    loader = RetrieverDataset(
        "wikimultihopqa",
        "wikimultihopqa-corpus",
        "config.ini",
        Split.DEV,
        tokenizer=None,
    )

    queries, qrels, corpus = loader.qrels()

    corpus_index = _build_corpus_index(corpus)

    # Initialize ADORE retriever
    retriever = AdoreRetriever(
        model_path=model_path,
        passage_memmap_path=passage_memmap_path,
        batch_size=32,
        device="cuda",
    )

    results = retriever.retrieve(
        corpus=corpus,
        queries=queries,
        top_k=k,
    )

    output: List[Dict[str, Any]] = []

    for q in queries:
        qid = q.id()
        qtext = q.text()

        if qid not in results:
            continue

        ranked = sorted(
            results[qid].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:k]

        contexts = []
        for doc_id, score in ranked:
            ev = corpus_index.get(doc_id)
            if ev is None:
                continue

            ctx = _evidence_to_dict(ev)
            ctx["score"] = float(score)
            contexts.append(ctx)

        output.append(
            {
                "id": qid,
                "question": qtext,
                "contexts": contexts,
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[adore_topk] Saved {len(output)} examples → {out_path}")