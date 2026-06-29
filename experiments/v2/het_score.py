"""Static, entropy-based heterogeneity scores + Otsu client classification.

Unlike the legacy ``heterogeneity.py`` (which derives scores from parameter
deltas observed during warm-up rounds), v2 computes a *static* het-score once,
before any training, from each client's known topic (cluster) distribution:

    p_k         = proportion of cluster k in client i's training data
    H_i         = -sum_k p_k * ln(p_k)
    H_max       = ln(num_clusters)         # num_clusters over the whole dataset
    het_score_i = 1 - H_i / H_max          # in [0, 1]; ->1 concentrated, ->0 uniform

Method 4 (HA-DualLoRA) uses the continuous score; Method 5 (Selective) uses the
Otsu binarization. Results are cached to ``results/het_scores_seed{seed}.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Tuple

from experiments.heterogeneity_aware_pfl.config import ExperimentConfig
from experiments.heterogeneity_aware_pfl.data_utils import (
    load_clustered_dataset,
    prepare_all_clients,
)
from experiments.heterogeneity_aware_pfl.heterogeneity import classify_clients_by_otsu


def _shannon_entropy(counts) -> float:
    """Natural-log Shannon entropy of a value-count Series."""
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        p = c / total
        if p > 0:
            h -= p * math.log(p)
    return h


def compute_het_scores(config: ExperimentConfig) -> Dict[str, object]:
    """Compute static het-scores and Otsu classification for all clients.

    Returns a JSON-serializable dict with het_scores, entropies, ground-truth and
    predicted client types, the Otsu threshold, and classification accuracy.
    """
    df, _ = load_clustered_dataset()
    num_clusters = int(df["cluster_id"].nunique())
    h_max = math.log(num_clusters)

    client_data = prepare_all_clients(config)
    # Exclude clients with no training data: an alpha=0.01 Dirichlet draw can
    # starve a client to 0 examples (seen only in extreme ablation splits, e.g.
    # 1 IID / 9 Non-IID). Such a client cannot participate and would skew the
    # Otsu threshold. No-op for the canonical 3/7 setting (no client is empty).
    client_data = {cid: cd for cid, cd in client_data.items() if len(cd["train"]) > 0}
    ground_truth = {cid: cd["type"] for cid, cd in client_data.items()}

    het_scores: Dict[int, float] = {}
    entropies: Dict[int, float] = {}
    for cid, cd in client_data.items():
        counts = cd["train"].to_pandas()["cluster_id"].value_counts()
        h_i = _shannon_entropy(counts)
        entropies[cid] = h_i
        het_scores[cid] = 1.0 - (h_i / h_max if h_max > 0 else 0.0)

    predicted, threshold, accuracy = classify_clients_by_otsu(het_scores, ground_truth)

    return {
        "seed": config.fl.seed,
        "num_clusters": num_clusters,
        "h_max": h_max,
        "het_scores": {str(c): het_scores[c] for c in sorted(het_scores)},
        "entropies": {str(c): entropies[c] for c in sorted(entropies)},
        "ground_truth": {str(c): ground_truth[c] for c in sorted(ground_truth)},
        "classification": {str(c): predicted[c] for c in sorted(predicted)},
        "otsu_threshold": threshold,
        "otsu_accuracy": accuracy,
    }


def het_scores_path(config: ExperimentConfig) -> Path:
    """Cache path for a run's het-scores, keyed by its own ``save_dir`` + seed.

    Main runs (``save_dir == RESULTS_DIR``) resolve to
    ``results/het_scores_seed{seed}.json`` exactly as before. Ablation runs cache
    under their own setting dir, so a changed IID/Non-IID split never reads the
    main run's stale scores (or another setting's).
    """
    return Path(config.save_dir) / f"het_scores_seed{config.fl.seed}.json"


def load_or_compute(config: ExperimentConfig) -> Dict[str, object]:
    """Load cached het-scores for this run, or compute and cache them."""
    path = het_scores_path(config)
    if path.exists():
        with open(path) as f:
            return json.load(f)
    result = compute_het_scores(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return result


def get_scores_and_classification(
    config: ExperimentConfig,
) -> Tuple[Dict[int, float], Dict[int, str]]:
    """Convenience: int-keyed het_scores and Otsu classification for a run."""
    data = load_or_compute(config)
    het = {int(c): float(v) for c, v in data["het_scores"].items()}
    cls = {int(c): str(v) for c, v in data["classification"].items()}
    return het, cls


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute v2 static het-scores")
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    from experiments.v2.v2_config import build_config

    config = build_config("het_score", args.seed)
    result = compute_het_scores(config)
    path = het_scores_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nhet-scores (seed {args.seed}) -> {path}")
    print(f"  num_clusters={result['num_clusters']}  H_max={result['h_max']:.3f}  "
          f"Otsu thr={result['otsu_threshold']:.4f}  acc={result['otsu_accuracy']:.2f}")
    for c in result["het_scores"]:
        gt = result["ground_truth"][c]
        pred = result["classification"][c]
        flag = "" if gt == pred else "  <-- misclassified"
        print(f"  client {c:>2} ({gt:6s}): H={result['entropies'][c]:.3f}  "
              f"het={result['het_scores'][c]:.4f}  pred={pred}{flag}")


if __name__ == "__main__":
    main()
