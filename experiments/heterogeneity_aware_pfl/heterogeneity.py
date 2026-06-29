"""Heterogeneity scoring and Otsu's method classification."""

from typing import Dict, List, Optional, Tuple

import numpy as np


def compute_client_delta(
    pre_params: List[np.ndarray], post_params: List[np.ndarray]
) -> np.ndarray:
    """Compute flattened parameter delta: post - pre."""
    pre_flat = np.concatenate([p.ravel() for p in pre_params])
    post_flat = np.concatenate([p.ravel() for p in post_params])
    return (post_flat - pre_flat).astype(np.float32)


def cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    u = u.ravel().astype(np.float64)
    v = v.ravel().astype(np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 1.0
    return float(1.0 - np.dot(u, v) / (nu * nv))


def compute_heterogeneity_scores(
    client_deltas: Dict[int, np.ndarray],
    global_delta: np.ndarray,
    client_sizes: Optional[Dict[int, int]] = None,
) -> Dict[int, float]:
    """Compute per-client het scores using leave-one-out aggregate as reference.

    Without client_sizes: legacy direct cosine_distance(delta_i, global_delta).
    With client_sizes: LOO reference removes self-domination bias —
        w_i = client_sizes[i] / total_examples
        loo_delta_i = (global_delta - w_i * delta_i) / (1 - w_i)
        het_score_i = cosine_distance(delta_i, loo_delta_i)
    """
    if client_sizes is None:
        return {
            cid: cosine_distance(delta, global_delta)
            for cid, delta in client_deltas.items()
        }
    total = float(sum(client_sizes.values()))
    scores = {}
    for cid, delta in client_deltas.items():
        w_i = client_sizes[cid] / total
        if 1.0 - w_i < 1e-12:
            scores[cid] = cosine_distance(delta, global_delta)
        else:
            loo_delta = (
                global_delta.astype(np.float64) - w_i * delta.astype(np.float64)
            ) / (1.0 - w_i)
            scores[cid] = cosine_distance(delta, loo_delta)
    return scores


def otsu_threshold(scores: List[float]) -> float:
    """Compute Otsu's threshold to binarize heterogeneity scores.

    With only 10 values, uses exhaustive search over all midpoints between
    sorted adjacent values to find the threshold minimizing total
    within-class variance.
    """
    sorted_scores = sorted(scores)
    n = len(sorted_scores)
    if n <= 1:
        return sorted_scores[0] if sorted_scores else 0.0

    best_threshold = sorted_scores[0]
    best_variance = float("inf")

    # Try every midpoint between adjacent sorted values
    for i in range(n - 1):
        t = (sorted_scores[i] + sorted_scores[i + 1]) / 2.0
        class0 = [s for s in scores if s <= t]
        class1 = [s for s in scores if s > t]

        if len(class0) == 0 or len(class1) == 0:
            continue

        w0 = len(class0) / n
        w1 = len(class1) / n
        var0 = np.var(class0)
        var1 = np.var(class1)
        within_class_var = w0 * var0 + w1 * var1

        if within_class_var < best_variance:
            best_variance = within_class_var
            best_threshold = t

    return best_threshold


def classify_clients_by_otsu(
    het_scores: Dict[int, float],
    ground_truth: Dict[int, str],
) -> Tuple[Dict[int, str], float, float]:
    """Apply Otsu's method to classify clients as 'iid' or 'noniid'.

    Returns: (classification_dict, threshold, accuracy)
    """
    scores_list = list(het_scores.values())
    threshold = otsu_threshold(scores_list)

    predicted = {}
    for cid, score in het_scores.items():
        predicted[cid] = "noniid" if score > threshold else "iid"

    # Compute accuracy against ground truth
    correct = sum(1 for cid in predicted if predicted[cid] == ground_truth[cid])
    accuracy = correct / len(predicted) if predicted else 0.0

    return predicted, threshold, accuracy


def heterogeneity_score_to_strength(
    avg_scores: Dict[int, float],
    lo_pct: float = 10.0,
    hi_pct: float = 90.0,
) -> Dict[int, float]:
    """Map raw heterogeneity scores to a per-client personalization strength in [0, 1].

    Uses a robust min-max normalization between the ``lo_pct`` and ``hi_pct``
    percentiles of the observed scores: values at or below ``lo_pct`` map to
    0.0, values at or above ``hi_pct`` map to 1.0, with linear interpolation
    in between. Intended to replace the hard Otsu threshold so borderline
    clients degrade gracefully instead of being hard-gated wrong.
    """
    if not avg_scores:
        return {}
    vals = np.asarray(list(avg_scores.values()), dtype=np.float64)
    lo = float(np.percentile(vals, lo_pct))
    hi = float(np.percentile(vals, hi_pct))
    if hi - lo < 1e-6:
        return {cid: 0.0 for cid in avg_scores}
    return {
        cid: float(np.clip((s - lo) / (hi - lo), 0.0, 1.0))
        for cid, s in avg_scores.items()
    }
