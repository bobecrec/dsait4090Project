"""
Evaluate experiment results with Exact Match and Cover Exact Match metrics.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

from src.evaluation_metrics import exact_match, cover_exact_match


def evaluate_file(file_path: str) -> Dict[str, Any]:
    """Evaluate a single results file."""
    with open(file_path, "r") as f:
        results = json.load(f)

    em_scores = []
    cover_em_scores = []

    for item in results:
        prediction = item.get("prediction", "")
        gold = item.get("gold_answer", "")

        em = exact_match(prediction, gold)
        cover_em = cover_exact_match(prediction, gold)

        em_scores.append(em)
        cover_em_scores.append(cover_em)

    n = len(results)
    avg_em = sum(em_scores) / n if n > 0 else 0
    avg_cover_em = sum(cover_em_scores) / n if n > 0 else 0

    return {
        "file": file_path,
        "num_samples": n,
        "exact_match": avg_em,
        "cover_exact_match": avg_cover_em,
    }


def print_results(metrics: Dict[str, Any]) -> None:
    """Pretty print evaluation results."""
    print(f"\n{'='*60}")
    print(f"File: {metrics['file']}")
    print(f"{'='*60}")
    print(f"  Samples:          {metrics['num_samples']}")
    print(f"  Exact Match:      {metrics['exact_match']:.4f} ({metrics['exact_match']*100:.2f}%)")
    print(f"  Cover Exact Match:{metrics['cover_exact_match']:.4f} ({metrics['cover_exact_match']*100:.2f}%)")


def evaluate_all():
    """Evaluate all available experiment results."""
    results_dir = Path("experiment_results")

    # Find all JSON result files
    result_files = []

    # Oracle
    oracle_path = results_dir / "oracle_gold_only.json"
    if oracle_path.exists():
        result_files.append(str(oracle_path))

    # Baseline top-k
    for k in [1, 3, 5]:
        baseline_path = results_dir / f"baseline_top{k}.json"
        if baseline_path.exists():
            result_files.append(str(baseline_path))

    # Random negatives
    random_dir = results_dir / "random_negatives"
    if random_dir.exists():
        for f in sorted(random_dir.glob("*.json")):
            result_files.append(str(f))

    # Hard negatives
    hard_dir = results_dir / "hard_negatives"
    if hard_dir.exists():
        for f in sorted(hard_dir.glob("*.json")):
            result_files.append(str(f))

    if not result_files:
        print("No result files found in experiment_results/")
        return

    print("\n" + "="*60)
    print("EVALUATION RESULTS SUMMARY")
    print("="*60)

    all_metrics = []
    for file_path in result_files:
        metrics = evaluate_file(file_path)
        print_results(metrics)
        all_metrics.append(metrics)

    # Summary table
    print("\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Experiment':<40} {'EM':>10} {'Cover EM':>12}")
    print("-"*62)
    for m in all_metrics:
        name = Path(m['file']).stem
        print(f"{name:<40} {m['exact_match']*100:>9.2f}% {m['cover_exact_match']*100:>11.2f}%")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Evaluate specific file
        metrics = evaluate_file(sys.argv[1])
        print_results(metrics)
    else:
        # Evaluate all
        evaluate_all()
