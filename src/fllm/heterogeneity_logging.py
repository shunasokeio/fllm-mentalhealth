"""
heterogeneity_logging.py

Per-client, per-round heterogeneity metrics for federated LoRA training.
Appends rows to a CSV file after each aggregation round without altering
any training logic.

Metrics
-------
round                       FL round number
client_id                   Flower client CID
cosine_distance_from_global 1 - cosine_similarity(client_delta, global_delta)
                            where delta = post_round_params - pre_round_params
mean_effective_rank_B       mean effective rank across all LoRA B matrices of
                            the client's post-round parameters.
                            effective_rank = exp(H),
                            H = -sum(p_i * log(p_i)),  p_i = sigma_i / sum(sigma)

LoRA B matrices are identified by shape (out_dim, lora_rank) where
shape[1] == lora_rank and shape[0] != lora_rank.
"""

import csv
import gc
import os
from typing import List, Optional

import numpy as np
import torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg


def _effective_rank(matrix: np.ndarray) -> float:
    """Effective rank = exp(H) where H is the entropy of normalised singular values."""
    mat = matrix.reshape(matrix.shape[0], -1).astype(np.float32)
    s = np.linalg.svd(mat, compute_uv=False)
    s = s[s > 1e-10]
    if len(s) == 0:
        return 1.0
    p = s / s.sum()
    H = -float(np.sum(p * np.log(p)))
    return float(np.exp(H))


def _cosine_distance(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    u = u.ravel().astype(np.float64)
    v = v.ravel().astype(np.float64)
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-12 or nv < 1e-12:
        return 1.0
    return float(1.0 - np.dot(u, v) / (nu * nv))


class FedAvgWithHeterogeneityLog(FedAvg):
    """FedAvg that logs per-client cosine distance and mean effective B-matrix rank.

    Parameters
    ----------
    log_path : str
        Path to the output CSV file.
    lora_rank : int
        LoRA rank r.  B matrices are identified as 2-D arrays whose second
        dimension equals lora_rank and whose first dimension does not.
    All other arguments are forwarded to FedAvg.
    """

    def __init__(
        self,
        *args,
        log_path: str = "heterogeneity_log.csv",
        lora_rank: int = 32,
        num_iid: int = 0,
        num_pathological: int = 0,
        starting_round: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._log_path = log_path
        self._lora_rank = lora_rank
        self._num_iid = num_iid
        self._num_pathological = num_pathological
        self._starting_round = starting_round  # offset for OOM-restart round continuity
        self._pre_round_flat: Optional[np.ndarray] = None
        self._wrote_header = os.path.exists(log_path)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _to_flat(self, ndarrays: List[np.ndarray]) -> np.ndarray:
        return np.concatenate([a.ravel().astype(np.float32) for a in ndarrays])

    def _is_lora_B(self, arr: np.ndarray) -> bool:
        r = self._lora_rank
        return arr.ndim == 2 and arr.shape[1] == r and arr.shape[0] != r

    # ------------------------------------------------------------------
    # strategy hooks
    # ------------------------------------------------------------------

    def configure_fit(self, server_round, parameters, client_manager):
        """Capture pre-round global parameters before they are sent to clients."""
        ndarrays = parameters_to_ndarrays(parameters)
        self._pre_round_flat = self._to_flat(ndarrays)
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        """Run normal FedAvg aggregation, then log heterogeneity metrics."""
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is None or aggregated[0] is None or self._pre_round_flat is None:
            return aggregated

        agg_params, agg_metrics = aggregated
        agg_flat = self._to_flat(parameters_to_ndarrays(agg_params))
        global_delta = agg_flat - self._pre_round_flat

        global_round = server_round + self._starting_round  # continuous across OOM restarts

        rows = []
        for client_proxy, fit_res in results:
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_delta = self._to_flat(client_ndarrays) - self._pre_round_flat

            cos_dist = _cosine_distance(client_delta, global_delta)

            b_matrices = [arr for arr in client_ndarrays if self._is_lora_B(arr)]
            mean_eff_rank = (
                float(np.mean([_effective_rank(B) for B in b_matrices]))
                if b_matrices
                else float("nan")
            )

            # partition_id from client metrics is the stable data-partition identifier;
            # client_proxy.cid is Flower's internal virtual-node ID which can be
            # non-sequential or reset after OOM restarts, so we don't rely on it.
            partition_id = int(fit_res.metrics.get("partition_id", -1))
            if self._num_iid > 0 and partition_id >= 0:
                if partition_id < self._num_iid:
                    partition_type = "iid"
                elif partition_id < self._num_iid + self._num_pathological:
                    partition_type = "pathological"
                else:
                    partition_type = "realistic"
            else:
                partition_type = "unknown"

            rows.append(
                {
                    "round": global_round,
                    "client_id": partition_id,  # stable data-partition ID, not Flower CID
                    "partition_id": partition_id,
                    "partition_type": partition_type,
                    "cosine_distance_from_global": cos_dist,
                    "mean_effective_rank_B": mean_eff_rank,
                }
            )

        _fields = [
            "round",
            "client_id",
            "partition_id",
            "partition_type",
            "cosine_distance_from_global",
            "mean_effective_rank_B",
        ]
        with open(self._log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_fields)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerows(rows)

        return aggregated


class FedAvgCoverageAware(FedAvgWithHeterogeneityLog):
    """FedAvg with softmax-weighted coverage-aware aggregation.

    Weights: p_k = softmax(γ * d_k) * |D_k| / Σ(softmax(γ * d_j) * |D_j|)
    where d_k = cosine_distance(client_k_delta, EMA_of_global_delta).
    γ=0 recovers standard FedAvg exactly.

    The reference vector for cosine distance is an exponential moving average
    (EMA) of the global aggregated update across rounds rather than the
    current round's update, which reduces noise in the heterogeneity signal.
    """

    def __init__(self, *args, gamma: float = 0.0, ema_path: Optional[str] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self._ema_beta: float = 0.9
        self._ema_path: Optional[str] = ema_path

        # Reload EMA from disk if available (survives OOM restarts).
        if ema_path and os.path.exists(ema_path):
            self._global_update_ema: Optional[np.ndarray] = np.load(ema_path)
            print(f"[CoverageAware] Loaded EMA from {ema_path}")
        else:
            self._global_update_ema = None

    def aggregate_fit(self, server_round, results, failures):
        # Step 1: Standard FedAvg aggregation + heterogeneity CSV logging (parent handles both)
        standard_aggregated, agg_metrics = super().aggregate_fit(server_round, results, failures)

        if standard_aggregated is None or not results:
            return standard_aggregated, agg_metrics

        # Step 2: Compute global_delta from standard FedAvg result and update EMA.
        # _pre_round_flat is set in configure_fit() before clients train.
        std_flat = self._to_flat(parameters_to_ndarrays(standard_aggregated))
        global_delta = std_flat - self._pre_round_flat

        if self._global_update_ema is None:
            self._global_update_ema = global_delta.copy()
        else:
            self._global_update_ema = (
                self._ema_beta * self._global_update_ema
                + (1 - self._ema_beta) * global_delta
            )

        if self._ema_path:
            np.save(self._ema_path, self._global_update_ema)

        # Free large intermediate arrays before continuing
        del std_flat, global_delta
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if self.gamma == 0.0:
            return standard_aggregated, agg_metrics

        # Step 3: Per-client cosine distances against the EMA reference vector.
        global_round = server_round + self._starting_round
        distances = []
        num_examples_list = []
        client_ndarrays_list = []
        partition_ids = []
        for _, fit_res in results:
            client_ndarrays = parameters_to_ndarrays(fit_res.parameters)
            client_delta = self._to_flat(client_ndarrays) - self._pre_round_flat
            distances.append(_cosine_distance(client_delta, self._global_update_ema))
            num_examples_list.append(fit_res.num_examples)
            client_ndarrays_list.append(client_ndarrays)
            partition_ids.append(int(fit_res.metrics.get("partition_id", -1)))

        distances = np.array(distances, dtype=np.float64)
        num_examples = np.array(num_examples_list, dtype=np.float64)

        # Debug: print per-client d_k values at rounds 1 and 10 to verify IID/non-IID separability.
        if global_round in (1, 10):
            print(f"\n[CoverageAware EMA debug] global_round={global_round}")
            for pid, dk in zip(partition_ids, distances):
                if self._num_iid > 0 and pid >= 0:
                    ptype = "iid" if pid < self._num_iid else "non-iid"
                else:
                    ptype = "unknown"
                print(f"  client {pid:3d} ({ptype}): d_k = {dk:.6f}")

        # Step 4: softmax(γ * d_k) * |D_k|, normalised
        gamma_d = self.gamma * distances
        gamma_d -= gamma_d.max()          # numerical stability
        softmax_w = np.exp(gamma_d)
        softmax_w /= softmax_w.sum()

        combined = softmax_w * num_examples
        combined /= combined.sum()        # sum-to-1 normalisation

        # Step 5: Re-aggregate with coverage-aware weights
        n_layers = len(client_ndarrays_list[0])
        custom_ndarrays = [
            sum(combined[k] * client_ndarrays_list[k][i]
                for k in range(len(combined)))
            for i in range(n_layers)
        ]

        print(
            f"[CoverageAware R{server_round}] gamma={self.gamma} | "
            f"distances(EMA)={distances.round(4).tolist()} | "
            f"weights={combined.round(4).tolist()} | "
            f"sum={combined.sum():.6f}"
        )

        del distances, num_examples, softmax_w, combined, client_ndarrays_list
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return ndarrays_to_parameters(custom_ndarrays), agg_metrics
