"""
Quick Random Negatives Experiment (Compressed Version)

Uses pre-computed baseline_top_100.json to avoid Contriever re-run.
Samples "random" negatives from other questions' retrieved docs.
Runs on subset of questions for faster results.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Any

from src.generative_model_setup import generate_answer
from src.evaluation_metrics import exact_match, cover_exact_match


# Set seed for reproducibility
random.seed(42)


def load_gold_answers(dev_path: str = "restricted_data/dev.json") -> Dict[str, str]:
    """Load gold answers from dev.json."""
    with open(dev_path, "r") as f:
        data = json.load(f)
    return {item["_id"]: item["answer"] for item in data}


def build_random_pool(all_data: List[Dict], exclude_qid: str) -> List[Dict]:
    """
    Build pool of random contexts from other questions.
    This simulates random negatives without needing the full corpus.
    """
    pool = []
    for item in all_data:
        if item["id"] != exclude_qid:
            # Take bottom-ranked docs (less relevant even to their own query)
            pool.extend(item["contexts"][50:100])  # Use ranks 50-100
    return pool


def run_random_negatives_quick(
    num_questions: int = 200,
    total_contexts: int = 5,
    noise_ratios: List[float] = [0.0, 0.5, 1.0],
    input_path: str = "experiment_results/baseline_top_100.json",
    out_dir: str = "experiment_results/random_negatives",
) -> Dict[str, Dict]:
    """
    Quick random negatives experiment using pre-computed retrieval.

    Args:
        num_questions: Number of questions to process (for speed)
        total_contexts: Total contexts per question
        noise_ratios: Ratios to test (0.0 = all relevant, 1.0 = all random)
        input_path: Pre-computed baseline results
        out_dir: Output directory
    """
    print(f"Loading pre-computed retrieval from {input_path}...")
    with open(input_path, "r") as f:
        all_data = json.load(f)

    # Limit to num_questions
    data = all_data[:num_questions]
    print(f"Using {len(data)} questions (of {len(all_data)} available)")

    # Load gold answers
    gold_answers = load_gold_answers()

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    all_metrics = {}

    for noise_ratio in noise_ratios:
        num_random = int(total_contexts * noise_ratio)
        num_relevant = total_contexts - num_random

        print(f"\n{'='*50}")
        print(f"[quick_random_neg] noise_ratio={noise_ratio:.0%}")
        print(f"  Relevant: {num_relevant}, Random: {num_random}")
        print(f"{'='*50}")

        output: List[Dict[str, Any]] = []
        em_scores = []
        cover_em_scores = []

        for i, item in enumerate(data):
            qid = item["id"]
            question = item["question"]
            gold_answer = gold_answers.get(qid, "")

            # Get top-k relevant contexts
            relevant_contexts = []
            for ctx in item["contexts"][:num_relevant]:
                relevant_contexts.append({
                    "doc_id": ctx["doc_id"],
                    "title": ctx["title"],
                    "text": ctx["text"],
                    "score": ctx["score"],
                    "is_random": False,
                })

            # Sample random contexts from other questions
            random_contexts = []
            if num_random > 0:
                pool = build_random_pool(all_data, qid)
                sampled = random.sample(pool, min(num_random, len(pool)))
                for ctx in sampled:
                    random_contexts.append({
                        "doc_id": ctx["doc_id"],
                        "title": ctx["title"],
                        "text": ctx["text"],
                        "score": 0.0,
                        "is_random": True,
                    })

            # Combine and shuffle
            all_contexts = relevant_contexts + random_contexts
            random.shuffle(all_contexts)

            # Generate answer
            prediction = generate_answer(question, all_contexts)

            # Compute metrics
            em = exact_match(prediction, gold_answer)
            cem = cover_exact_match(prediction, gold_answer)
            em_scores.append(em)
            cover_em_scores.append(cem)

            output.append({
                "id": qid,
                "question": question,
                "gold_answer": gold_answer,
                "noise_ratio": noise_ratio,
                "num_relevant": len(relevant_contexts),
                "num_random": len(random_contexts),
                "contexts": all_contexts,
                "prediction": prediction,
                "exact_match": em,
                "cover_exact_match": cem,
            })

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(data)} questions...")

        # Calculate metrics
        avg_em = sum(em_scores) / len(em_scores) if em_scores else 0
        avg_cem = sum(cover_em_scores) / len(cover_em_scores) if cover_em_scores else 0

        all_metrics[noise_ratio] = {
            "exact_match": avg_em,
            "cover_exact_match": avg_cem,
        }

        # Save results
        out_path = f"{out_dir}/random_noise_{int(noise_ratio*100)}pct.json"
        result_output = {
            "config": {
                "noise_ratio": noise_ratio,
                "num_relevant": num_relevant,
                "num_random": num_random,
                "num_questions": len(data),
                "total_contexts": total_contexts,
            },
            "metrics": {
                "exact_match": avg_em,
                "cover_exact_match": avg_cem,
            },
            "results": output,
        }

        with open(out_path, "w") as f:
            json.dump(result_output, f, indent=2)

        print(f"\n  Results: EM={avg_em*100:.2f}%, Cover EM={avg_cem*100:.2f}%")
        print(f"  Saved to: {out_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY: Random Negatives Quick Experiment")
    print(f"{'='*60}")
    print(f"{'Noise %':<10} {'EM':>12} {'Cover EM':>12}")
    print("-" * 35)
    for ratio, metrics in sorted(all_metrics.items()):
        print(f"{ratio*100:>6.0f}%    {metrics['exact_match']*100:>10.2f}%  {metrics['cover_exact_match']*100:>10.2f}%")

    return all_metrics


if __name__ == "__main__":
    run_random_negatives_quick(
        num_questions=200,
        noise_ratios=[0.0, 0.5, 1.0],
    )
