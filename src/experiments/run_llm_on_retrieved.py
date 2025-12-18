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
import random
from pathlib import Path
from typing import Dict, List, Any

from src.generative_model_setup import generate_answer
from src.evaluation_metrics import exact_match, cover_exact_match


def load_gold_answers(dev_path: str = "restricted_data/dev.json") -> Dict[str, str]:
    """Load gold answers from dev.json, keyed by question ID."""
    with open(dev_path, "r") as f:
        data = json.load(f)
    return {item["_id"]: item["answer"] for item in data}

def load_gold_titles(dev_path: str = "restricted_data/dev.json") -> Dict[str, set]:
    """Map question id to set of gold document titles."""
    with open(dev_path, "r") as f:
        data = json.load(f)
    qid_to_titles = {}
    for item in data:
        qid = item["_id"]
        titles = {sf[0].lower() for sf in item.get("supporting_facts", [])}
        qid_to_titles[qid] = titles
    return qid_to_titles


def split_gold_and_non_gold_contexts(retrieved: List[Dict[str, Any]], gold_titles: set,):
    """Split a ranked list of retrieved contexts into gold and non-gold subsets."""
    gold_ctx = []
    non_gold_ctx = []
    for ctx in retrieved:
        title = (ctx.get("title") or "").lower()
        if title in gold_titles:
            gold_ctx.append(ctx)
        else:
            non_gold_ctx.append(ctx)
    return gold_ctx, non_gold_ctx

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

def run_llm_with_hard_negatives(
    k_gold: int,
    k_hard: int,
    mix_topk: int = 0,
    input_path: str = "experiment_results/baseline_top_100.json",
    output_path: str | None = None,
) -> Dict[str, float]:
    """Run LLM with gold contexts + hard negatives and optional baseline top-k contexts"""
    if output_path is None:
        suffix = f"gold{k_gold}_hard{k_hard}"
        if mix_topk > 0:
            suffix += f"_top{mix_topk}"
        output_path = f"experiment_results/hard_negatives/{suffix}_scored.json"

    print(f"\n{'='*60}")
    print(f"Running LLM with hard negatives: gold={k_gold}, hard={k_hard}, topk={mix_topk}")
    print(f"{'='*60}")

    # Load retrieval and labels
    with open(input_path, "r") as f:
        data = json.load(f)
    gold_answers = load_gold_answers()
    gold_titles_map = load_gold_titles()

    results = []
    em_scores = []
    cover_em_scores = []

    print(f"Processing {len(data)} queries...")
    for i, item in enumerate(data):
        qid = item["id"]
        question = item["question"]
        retrieved = item["contexts"]

        gold_titles = gold_titles_map.get(qid, set())
        gold_ctx_all, non_gold_ctx = split_gold_and_non_gold_contexts(retrieved, gold_titles)

        gold_ctx = gold_ctx_all[:k_gold] if k_gold > 0 else []
        hard_ctx = non_gold_ctx[:k_hard] if k_hard > 0 else []
        extra_topk = retrieved[:mix_topk] if mix_topk > 0 else []

        contexts = gold_ctx + hard_ctx + extra_topk
        random.shuffle(contexts)

        gold_answer = gold_answers.get(qid, "")
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
            "k_gold": k_gold,
            "k_hard": k_hard,
            "mix_topk": mix_topk,
        })

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(data)} queries...")

    avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
    avg_cem = sum(cover_em_scores) / len(cover_em_scores) if cover_em_scores else 0

    output = {
        "config": {
            "k_gold": k_gold,
            "k_hard": k_hard,
            "mix_topk": mix_topk,
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

    print(f"\n{'='*60}")
    print(f"HARD-NEG RESULTS: gold={k_gold}, hard={k_hard}, topk={mix_topk}")
    print(f"{'='*60}")
    print(f"  Exact Match:       {avg_em:.4f} ({avg_em*100:.2f}%)")
    print(f"  Cover Exact Match: {avg_cem:.4f} ({avg_cem*100:.2f}%)")
    print(f"  Saved to: {output_path}")

    return {"exact_match": avg_em, "cover_exact_match": avg_cem}

def run_all_hard_negative_configs():
    configs = [
        (5, 0, 0), # all gold
        (4, 1, 0), # 1 hard out of 5
        (3, 2, 0),
        (0, 5, 0), # all hard
        (3, 2, 5), # 3 hard + top-5 baseline
    ]
    summary = {}
    for k_gold, k_hard, mix_topk in configs:
        key = f"gold{k_gold}_hard{k_hard}_top{mix_topk}"
        summary[key] = run_llm_with_hard_negatives(
            k_gold=k_gold,
            k_hard=k_hard,
            mix_topk=mix_topk,
        )
    return summary


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "hard":
        # Run hard negatives configurations
        run_all_hard_negative_configs()
    elif len(sys.argv) > 1:
        # Run specific k
        k = int(sys.argv[1])
        run_llm_on_baseline(k)
    else:
        # Run all
        run_all_baselines()
