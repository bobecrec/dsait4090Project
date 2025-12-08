import json
from pathlib import Path
from typing import Dict, List, Any

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


def run_baseline_topk(k: int, out_path: str) -> None:
    """
      "Use off-the-shelf Contriever to get top-k contexts per query,
      feed them to a generative model, and save predictions."
    """
    print(f"[baseline_topk] Running Contriever with k_retrieval={k} ...")
    queries, qrels, corpus, results = run_contriever(k_retrieval=k)

    corpus_index = _build_corpus_index(corpus)
    output: List[Dict[str, Any]] = []

    # Loop over all dev.json
    for q in queries:
        qid = q.id()
        qtext = q.text()

        if qid not in results:
            continue

        # sort retrieved docs by score desc
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

        # call generative model
        answer = generate_answer(qtext, contexts)

        output.append(
            {
                "id": qid,
                "question": qtext,
                "contexts": contexts,
                "prediction": answer,
            }
        )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[baseline_topk] Saved {len(output)} examples → {out_path}")


def run_all_ks():
    base_out = Path("experiment_results")
    for k in (1, 3, 5):
        out_path = base_out / f"baseline_top{k}.json"
        run_baseline_topk(k=k, out_path=str(out_path))