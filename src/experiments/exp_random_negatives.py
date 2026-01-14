"""
Experiment: Random Negatives

Mix top-k retrieved contexts with randomly sampled documents from the corpus.
Analyze how random noise affects QA performance.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any, Set

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


def sample_random_docs(
    corpus_index: Dict,
    exclude_ids: Set[str],
    num_docs: int,
) -> List[Dict[str, Any]]:
    """Sample random documents from corpus, excluding specified IDs."""
    available_ids = [doc_id for doc_id in corpus_index.keys() if doc_id not in exclude_ids]

    if len(available_ids) < num_docs:
        sampled_ids = available_ids
    else:
        sampled_ids = random.sample(available_ids, num_docs)

    contexts = []
    for doc_id in sampled_ids:
        ev = corpus_index[doc_id]
        ctx = _evidence_to_dict(ev)
        ctx["score"] = 0.0  # Random docs have no retrieval score
        ctx["is_random"] = True
        contexts.append(ctx)

    return contexts


def run_random_negatives(
    total_contexts: int = 5,
    noise_ratios: List[float] = [0.0, 0.25, 0.5, 0.75, 1.0],
    out_dir: str = "experiment_results/random_negatives",
    k_retrieval: int = 100,  # Retrieve more to have buffer
) -> None:
    """
    Run random negatives experiment at different noise ratios.

    Args:
        total_contexts: Total number of contexts to provide to LLM
        noise_ratios: List of ratios (0.0 = all relevant, 1.0 = all random)
        out_dir: Directory to save results
        k_retrieval: Number of docs to retrieve from Contriever
    """
    print(f"[random_negatives] Running Contriever with k={k_retrieval}...")
    queries, qrels, corpus, results = run_contriever(k_retrieval=k_retrieval)

    corpus_index = _build_corpus_index(corpus)

    # Load dev.json to get supporting_facts for gold document identification
    with open("restricted_data/dev.json", "r") as f:
        dev_data = json.load(f)
    qid_to_sample = {s["_id"]: s for s in dev_data}

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for noise_ratio in noise_ratios:
        num_random = int(total_contexts * noise_ratio)
        num_relevant = total_contexts - num_random

        print(f"\n{'='*50}")
        print(f"[random_negatives] noise_ratio={noise_ratio:.0%}")
        print(f"  Relevant: {num_relevant}, Random: {num_random}")
        print(f"{'='*50}")

        output: List[Dict[str, Any]] = []

        for i, q in enumerate(queries):
            qid = q.id()
            qtext = q.text()
            sample = qid_to_sample.get(qid, {})
            gold_answer = sample.get("answer", "")

            if qid not in results:
                continue

            # Get gold doc IDs to exclude from random sampling
            gold_ids = get_gold_doc_ids(sample, corpus_index)

            # Get top relevant documents
            ranked = sorted(
                results[qid].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:num_relevant]

            relevant_contexts = []
            relevant_ids = set()
            for doc_id, score in ranked:
                ev = corpus_index.get(doc_id)
                if ev is None:
                    continue
                ctx = _evidence_to_dict(ev)
                ctx["score"] = float(score)
                ctx["is_random"] = False
                relevant_contexts.append(ctx)
                relevant_ids.add(doc_id)

            # Sample random documents (exclude gold and already retrieved)
            exclude_ids = gold_ids | relevant_ids
            random_contexts = sample_random_docs(corpus_index, exclude_ids, num_random)

            # Combine and shuffle
            all_contexts = relevant_contexts + random_contexts
            random.shuffle(all_contexts)

            # Generate answer
            prediction = generate_answer(qtext, all_contexts)

            output.append({
                "id": qid,
                "question": qtext,
                "gold_answer": gold_answer,
                "noise_ratio": noise_ratio,
                "num_relevant": len(relevant_contexts),
                "num_random": len(random_contexts),
                "contexts": all_contexts,
                "prediction": prediction,
            })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(queries)} questions...")

        # Save results for this ratio
        out_path = f"{out_dir}/random_noise_{int(noise_ratio*100)}pct.json"
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        print(f"[random_negatives] Saved {len(output)} examples -> {out_path}")


def run_all_ratios():
    """Run experiment with standard noise ratios."""
    run_random_negatives(
        total_contexts=5,
        noise_ratios=[0.0, 0.25, 0.5, 0.75, 1.0],
    )


if __name__ == "__main__":
    run_all_ratios()
