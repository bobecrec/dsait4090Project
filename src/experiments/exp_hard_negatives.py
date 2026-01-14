"""
Experiment: Hard Negatives

Mix gold contexts with hard negatives - documents that are semantically similar
to the query but do not help answer the question (not in ground truth).

Hard negatives are more challenging than random noise because they are
related to the topic but potentially misleading.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple

from src.retrieval.contriever_retrieval import run_contriever
from src.generative_model_setup import generate_answer

# Set seed for reproducibility
random.seed(42)


def _build_corpus_index(corpus) -> Dict[str, Any]:
    """doc_id -> Evidence object"""
    return {doc.id(): doc for doc in corpus}


def _evidence_to_dict(ev) -> Dict[str, str]:
    """Convert Evidence object to dict."""
    return {
        "doc_id": ev.id(),
        "title": ev.title() or "",
        "text": ev.text() or "",
    }


def get_gold_doc_ids(sample: Dict[str, Any], corpus_index: Dict) -> Set[str]:
    """Get document IDs that are in the gold set for this question."""
    gold_titles = set()
    for fact in sample.get("supporting_facts", []):
        if len(fact) >= 1:
            gold_titles.add(fact[0].lower())

    gold_ids = set()
    for doc_id, ev in corpus_index.items():
        title = (ev.title() or "").lower()
        for gold_title in gold_titles:
            if gold_title in title or title in gold_title:
                gold_ids.add(doc_id)

    return gold_ids


def get_hard_negatives(
    ranked_results: List[Tuple[str, float]],
    gold_ids: Set[str],
    corpus_index: Dict,
    num_hard_neg: int,
) -> List[Dict[str, Any]]:
    """
    Get hard negatives: high-scoring documents that are NOT in gold set.

    These are documents that Contriever thinks are relevant (high similarity)
    but don't actually help answer the question.
    """
    hard_negatives = []

    for doc_id, score in ranked_results:
        if doc_id in gold_ids:
            continue  # Skip gold documents

        ev = corpus_index.get(doc_id)
        if ev is None:
            continue

        ctx = _evidence_to_dict(ev)
        ctx["score"] = float(score)
        ctx["is_hard_negative"] = True
        hard_negatives.append(ctx)

        if len(hard_negatives) >= num_hard_neg:
            break

    return hard_negatives


def get_gold_contexts_from_retrieval(
    ranked_results: List[Tuple[str, float]],
    gold_ids: Set[str],
    corpus_index: Dict,
    max_gold: int,
) -> List[Dict[str, Any]]:
    """Get gold documents that appear in retrieval results."""
    gold_contexts = []

    for doc_id, score in ranked_results:
        if doc_id not in gold_ids:
            continue

        ev = corpus_index.get(doc_id)
        if ev is None:
            continue

        ctx = _evidence_to_dict(ev)
        ctx["score"] = float(score)
        ctx["is_hard_negative"] = False
        gold_contexts.append(ctx)

        if len(gold_contexts) >= max_gold:
            break

    return gold_contexts


def run_hard_negatives(
    total_contexts: int = 5,
    hard_neg_ratios: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    out_dir: str = "experiment_results/hard_negatives",
    k_retrieval: int = 100,
) -> None:
    """
    Run hard negatives experiment at different ratios.

    Args:
        total_contexts: Total number of contexts to provide to LLM
        hard_neg_ratios: Ratio of hard negatives (0.0 = all gold, 1.0 = all hard neg)
        out_dir: Directory to save results
        k_retrieval: Number of docs to retrieve for hard negative pool
    """
    print(f"[hard_negatives] Running Contriever with k={k_retrieval}...")
    queries, qrels, corpus, results = run_contriever(k_retrieval=k_retrieval)

    corpus_index = _build_corpus_index(corpus)

    # Load dev.json to get supporting_facts
    with open("restricted_data/dev.json", "r") as f:
        dev_data = json.load(f)
    qid_to_sample = {s["_id"]: s for s in dev_data}

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for hard_neg_ratio in hard_neg_ratios:
        num_hard_neg = int(total_contexts * hard_neg_ratio)
        num_gold = total_contexts - num_hard_neg

        print(f"\n{'='*50}")
        print(f"[hard_negatives] hard_neg_ratio={hard_neg_ratio:.0%}")
        print(f"  Gold: {num_gold}, Hard Negatives: {num_hard_neg}")
        print(f"{'='*50}")

        output: List[Dict[str, Any]] = []

        for i, q in enumerate(queries):
            qid = q.id()
            qtext = q.text()
            sample = qid_to_sample.get(qid, {})
            gold_answer = sample.get("answer", "")

            if qid not in results:
                continue

            # Get gold doc IDs
            gold_ids = get_gold_doc_ids(sample, corpus_index)

            # Sort all retrieved results by score
            ranked = sorted(
                results[qid].items(),
                key=lambda x: x[1],
                reverse=True,
            )

            # Get gold contexts from retrieval
            gold_contexts = get_gold_contexts_from_retrieval(
                ranked, gold_ids, corpus_index, num_gold
            )

            # Get hard negatives (high score but not gold)
            hard_neg_contexts = get_hard_negatives(
                ranked, gold_ids, corpus_index, num_hard_neg
            )

            # Combine and shuffle
            all_contexts = gold_contexts + hard_neg_contexts
            random.shuffle(all_contexts)

            # Generate answer
            prediction = generate_answer(qtext, all_contexts)

            output.append({
                "id": qid,
                "question": qtext,
                "gold_answer": gold_answer,
                "hard_neg_ratio": hard_neg_ratio,
                "num_gold": len(gold_contexts),
                "num_hard_negatives": len(hard_neg_contexts),
                "contexts": all_contexts,
                "prediction": prediction,
            })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(queries)} questions...")

        # Save results for this ratio
        out_path = f"{out_dir}/hard_neg_{int(hard_neg_ratio*100)}pct.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[hard_negatives] Saved {len(output)} examples -> {out_path}")


def run_all_ratios():
    """Run experiment with standard hard negative ratios."""
    run_hard_negatives(
        total_contexts=5,
        hard_neg_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
    )


if __name__ == "__main__":
    run_all_ratios()
