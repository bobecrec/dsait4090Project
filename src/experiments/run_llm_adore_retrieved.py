"""
Run LLM on already-retrieved ADORE contexts.

This script:
1. Reads adore_retrieval_top_50.json
2. Uses top-k contexts (k = 1, 3, 5)
3. Runs LLM to generate predictions
4. Gets gold answers from dev.json
5. Computes EM and Cover EM metrics
6. Saves results per k in experiment_results/adore/
"""

import json
from pathlib import Path
from typing import Dict, List

from src.generative_model_setup import generate_answer
from src.evaluation_metrics import exact_match, cover_exact_match


from typing import Dict, List
import json


def load_gold_answers_contexts(dev_path: str = "restricted_data/dev.json"):
    """
    Load gold answers and gold contexts from dev.json.

    Returns:
        answers: dict[qid -> answer]
        contexts: dict[qid -> List[{title, text}]]
    """
    with open(dev_path, "r") as f:
        data = json.load(f)

    answers: Dict[str, str] = {}
    final_contexts: Dict[str, List[Dict[str, str]]] = {}

    for item in data:
        qid = item["_id"]
        answers[qid] = item["answer"]

        query_contexts = []
        for title, sentences in item["context"]:
            text = " ".join(sentences)
            query_contexts.append({
                "title": title,
                "text": text,
            })
        final_contexts[qid] = query_contexts
    return answers, final_contexts



def run_llm_on_adore(
        k: int,
        input_path: str = "experiment_results/adore/adore_retrieval_top_50.json",
        output_path: str = None,
        gold: bool = False
) -> Dict[str, float]:
    """
    Run LLM on pre-retrieved ADORE contexts.

    Args:
        k: Number of contexts to use (1, 3, or 5)
        input_path: Path to adore retrieval file
        output_path: Path to save scored results

    Returns:
        Dictionary with EM and Cover EM scores
    """
    if output_path is None:
        output_path = f"experiment_results/adore/adore_top{k}_scored.json"

    print(f"\n{'=' * 60}")
    print(f"Running LLM on ADORE (top-{k})")
    print(f"{'=' * 60}")

    # Load retrieved contexts
    print(f"Loading contexts from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)

    # Load gold answers
    print("Loading gold answers from dev.json...")
    gold_answers, gold_contexts = load_gold_answers_contexts()

    results = []
    em_scores = []
    cover_em_scores = []

    print(f"Processing {len(data)} queries...")
    for i, item in enumerate(data):
        qid = item["id"]
        question = item["question"]
        contexts = item["contexts"][:k]

        gold_answer = gold_answers.get(qid, "")
        gold_contexts_query = gold_contexts.get(qid, "")[:3]

        if gold:
            for item in gold_contexts_query:
                contexts.append(item)

        prediction = generate_answer(question, contexts)

        em = exact_match(prediction, gold_answer)
        cem = cover_exact_match(prediction, gold_answer)

        em_scores.append(em)
        cover_em_scores.append(cem)

        results.append({
            "id": qid,
            "question": question,
            "contexts": contexts,
            "gold_answer": gold_answer,
            "prediction": prediction,
            "exact_match": em,
            "cover_exact_match": cem,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} queries...")

    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
    avg_cem = sum(cover_em_scores) / len(cover_em_scores) if cover_em_scores else 0

    output = {
        "config": {
            "retriever": "adore",
            "k": k,
            "num_queries": len(data),
            "input_file": input_path,
        },
        "metrics": {
            "exact_match": avg_em,
            "cover_exact_match": avg_cem,
        },
        "results": results,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"RESULTS for ADORE k={k}")
    print(f"{'=' * 60}")
    print(f"  Exact Match:       {avg_em:.4f} ({avg_em * 100:.2f}%)")
    print(f"  Cover Exact Match: {avg_cem:.4f} ({avg_cem * 100:.2f}%)")
    print(f"  Saved to: {output_path}")

    return {"exact_match": avg_em, "cover_exact_match": avg_cem}


def run_all_adore(gold):
    """Run LLM on ADORE retrieval for k=1,3,5."""
    all_metrics = {}

    for k in [1, 3, 5]:
        metrics = run_llm_on_adore(k, gold=True)
        all_metrics[k] = metrics

    print(f"\n{'=' * 60}")
    print("SUMMARY: ADORE Retrieval")
    print(f"{'=' * 60}")
    print(f"{'k':<5} {'EM':>12} {'Cover EM':>12}")
    print("-" * 30)
    for k, metrics in all_metrics.items():
        print(f"{k:<5} {metrics['exact_match'] * 100:>11.2f}% "
              f"{metrics['cover_exact_match'] * 100:>11.2f}%")

    return all_metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        k = int(sys.argv[1])
        run_llm_on_adore(k)
    else:
        run_all_adore(True)
    # gold_answers, gold_contexts = load_gold_answers_contexts()
    # print(type(gold_contexts))
    # print(type(gold_contexts.get("61a46987092f11ebbdaeac1f6bf848b6")))
    # print(type(gold_contexts.get("61a46987092f11ebbdaeac1f6bf848b6")[0]))
    # print(gold_contexts.get("61a46987092f11ebbdaeac1f6bf848b6")[0].keys())
