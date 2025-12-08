# src/main.py
import argparse

from src.experiments.exp_baseline_topk import run_all_ks, run_baseline_topk
# you can import other experiments: (this is only for the first bulletpoint)
# from src.experiments.exp_random_negatives import run_random_negatives
# from src.experiments.exp_hard_negatives import run_hard_negatives


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp",
        type=str,
        required=True,
        choices=["baseline_topk"],  # extend later
        help="Which experiment to run",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=None,
        help="k for top-k contexts (if applicable)",
    )
    args = parser.parse_args()

    if args.exp == "baseline_topk":
        if args.k is None:
            # run all three ks
            run_all_ks()
        else:
            out_path = f"experiment_results/baseline_top{args.k}.json"
            run_baseline_topk(k=args.k, out_path=out_path)


if __name__ == "__main__":
    main()