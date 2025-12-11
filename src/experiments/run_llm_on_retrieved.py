"""
Run LLM on already-retrieved contexts.

This script:
1. Reads baseline_top{k}.json (contexts already retrieved by Contriever)
2. Runs Claude to generate predictions
3. Gets gold answers from dev.json
4. Computes EM and Cover EM metrics
5. Saves complete results
"""

import json
from pathlib import Path
from typing import Dict, List, Any

from src.generative_model_setup import generate_answer
from src.evaluation_metrics import exact_match, cover_exact_match


def load_gold_answers(dev_path: str = "restricted_data/dev.json") -> Dict[str, str]:
    """Load gold answers from dev.json, keyed by question ID."""
    with open(dev_path, "r") as f:
        data = json.load(f)
    return {item["_id"]: item["answer"] for item in data}


def run_llm_on_baseline(
    k: int,
    input_path: str = None,
    output_path: str = None,
) -> Dict[str, float]:
    """
    Run LLM on pre-retrieved baseline contexts.

    Args:
        k: Number of contexts (1, 3, or 5)
        input_path: Path to baseline results file (default: experiment_results/baseline_top{k}.json)
        output_path: Path to save results (default: experiment_results/baseline_top{k}_scored.json)

    Returns:
        Dictionary with EM and Cover EM scores
    """
    if input_path is None:
        input_path = f"experiment_results/baseline_top{k}.json"
        # Fallback to baseline_top_100.json if specific file doesn't exist
        if not Path(input_path).exists():
            alt_path = "experiment_results/baseline_top_100.json"
            if Path(alt_path).exists():
                print(f"Note: {input_path} not found, using top-{k} from baseline_top_100.json")
                input_path = alt_path
    if output_path is None:
        output_path = f"experiment_results/baseline_top{k}_scored.json"

    print(f"\n{'='*60}")
    print(f"Running LLM on baseline_top{k}")
    print(f"{'='*60}")

    # Load retrieved contexts
    print(f"Loading contexts from {input_path}...")
    with open(input_path, "r") as f:
        data = json.load(f)

    # Load gold answers
    print("Loading gold answers from dev.json...")
    gold_answers = load_gold_answers()

    # Process each query
    results = []
    em_scores = []
    cover_em_scores = []

    print(f"Processing {len(data)} queries...")
    for i, item in enumerate(data):
        qid = item["id"]
        question = item["question"]
        contexts = item["contexts"][:k]  # Take top-k contexts

        # Get gold answer
        gold_answer = gold_answers.get(qid, "")

        # Generate prediction with LLM
        prediction = generate_answer(question, contexts)

        # Compute metrics
        em = exact_match(prediction, gold_answer)
        cem = cover_exact_match(prediction, gold_answer)

        em_scores.append(em)
        cover_em_scores.append(cem)

        # Save result
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

    # Calculate overall metrics
    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
    avg_cem = sum(cover_em_scores) / len(cover_em_scores) if cover_em_scores else 0

    # Add summary to output
    output = {
        "config": {
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

    # Save results
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS for k={k}")
    print(f"{'='*60}")
    print(f"  Exact Match:       {avg_em:.4f} ({avg_em*100:.2f}%)")
    print(f"  Cover Exact Match: {avg_cem:.4f} ({avg_cem*100:.2f}%)")
    print(f"  Saved to: {output_path}")

    return {"exact_match": avg_em, "cover_exact_match": avg_cem}


def run_all_baselines():
    """Run LLM on all baseline files (k=1, 3, 5)."""
    all_metrics = {}

    for k in [1, 3, 5]:
        input_path = f"experiment_results/baseline_top{k}.json"
        if Path(input_path).exists():
            metrics = run_llm_on_baseline(k)
            all_metrics[k] = metrics
        else:
            # Try baseline_top_100.json and use top-k from it
            alt_path = "experiment_results/baseline_top_100.json"
            if Path(alt_path).exists():
                print(f"\nbaseline_top{k}.json not found, using top-{k} from baseline_top_100.json")
                metrics = run_llm_on_baseline(k, input_path=alt_path)
                all_metrics[k] = metrics
            else:
                print(f"\nSkipping k={k}: {input_path} not found")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: All Baselines")
    print(f"{'='*60}")
    print(f"{'k':<5} {'EM':>12} {'Cover EM':>12}")
    print("-" * 30)
    for k, metrics in all_metrics.items():
        print(f"{k:<5} {metrics['exact_match']*100:>11.2f}% {metrics['cover_exact_match']*100:>11.2f}%")

    return all_metrics


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Run specific k
        k = int(sys.argv[1])
        run_llm_on_baseline(k)
    else:
        # Run all
        run_all_baselines()
