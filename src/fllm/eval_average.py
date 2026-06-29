"""
Load eval results JSON and compute the average of each metric.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict


def load_results(path: Path):
    """Load eval results: JSON array or dict with 'results' key."""
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "results" in data:
        return data["results"]
    raise ValueError("Expected a JSON array or object with 'results' key")


def average_metrics(results: list) -> dict:
    """
    Compute mean score for each metric across all results.
    Skips 'question' and 'response'; treats other int/float values as metrics.
    """
    by_metric = defaultdict(list)
    for item in results:
        for key, value in item.items():
            if key in ("question", "response"):
                continue
            if isinstance(value, (int, float)):
                by_metric[key].append(value)
    return {
        metric: sum(scores) / len(scores)
        for metric, scores in by_metric.items()
        if scores
    }


def main():
    parser = argparse.ArgumentParser(description="Compute average of each metric from eval results JSON.")
    parser.add_argument(
        "--input",
        default="eval_results7/FL_iid_qwen0.5b_newcluster_seed75.json",
        help="Path to eval results JSON (array of result objects)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional: path to save averages JSON (default: only print)",
    )
    args = parser.parse_args()

    path = Path(args.input)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    results = load_results(path)
    if not results:
        print("No results to average.")
        return

    means = average_metrics(results)
    print(f"Loaded {len(results)} results from {path}")
    print("\nAverage scores:")
    for metric in sorted(means.keys()):
        print(f"  {metric}: {means[metric]:.2f}")
    if means:
        means_excl = {k: v for k, v in means.items() if k != "Overall Rating"}
        print(means_excl)
        if means_excl:
            overall_mean = sum(means_excl.values()) / len(means_excl)
            print(f"\nMean of all {len(means_excl)} metrics (excluding overall_rating): {overall_mean:.2f}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(means, f, indent=2)
        print(f"\nAverages saved to {out_path}")


if __name__ == "__main__":
    main()
