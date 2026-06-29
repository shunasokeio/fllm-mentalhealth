"""v2 federated round loop + aggregation, and the per-run training entry point.

Run a single (method, seed) experiment:

    CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.server_v2 --method fedavg --seed 42

The server executes each client's :class:`ClientPlan` via the constant-LR
trainer, FedAvg-aggregates the uploaded global adapters (weighted by example
count), optionally broadcasts the result, logs per-round metrics to
``results/{run}/metrics.jsonl``, and finally persists adapters + a
``manifest.json`` that the evaluator uses to reconstruct each client's model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.heterogeneity_aware_pfl.config import ExperimentConfig
from experiments.heterogeneity_aware_pfl.data_utils import prepare_all_clients
from experiments.heterogeneity_aware_pfl.utils import ExperimentLogger, set_seed, setup_file_logging

from experiments.v2 import trainer
from experiments.v2.het_score import get_scores_and_classification
from experiments.v2.methods import REGISTRY, ClientContext, ClientPlan, MethodSpec, get_method
from experiments.v2.v2_config import CONFIGS_DIR, LOGS_DIR, build_config, dump_config_yaml

Params = List[np.ndarray]


def fedavg(updates: List[Tuple[Params, int]]) -> Params:
    """Example-weighted FedAvg over (params, num_examples) uploads."""
    total = float(sum(n for _, n in updates))
    agg = [np.zeros_like(p) for p in updates[0][0]]
    for params, n in updates:
        w = n / total
        for i, p in enumerate(params):
            agg[i] = agg[i] + w * p.astype(agg[i].dtype)
    return agg


class V2Server:
    def __init__(
        self,
        config: ExperimentConfig,
        spec: MethodSpec,
        het_scores: Dict[int, float],
        classification: Dict[int, str],
        gpu_id: int = 0,
    ):
        self.config = config
        self.spec = spec
        self.het = het_scores
        self.cls = classification
        self.gpu_id = gpu_id

        self.client_data = prepare_all_clients(config)
        self.client_types = {cid: cd["type"] for cid, cd in self.client_data.items()}
        self.logger = ExperimentLogger(config.save_dir, config.experiment_name)
        self.run_dir = Path(config.save_dir) / config.experiment_name

    def run(self) -> None:
        set_seed(self.config.fl.seed)
        self.logger.reset_round_logs()

        init = trainer.initial_global_params(self.config, gpu_id=self.gpu_id)
        global_params: Params = init
        client_primary: Dict[int, Params] = {cid: init for cid in self.client_data}
        client_local: Dict[int, Optional[Params]] = {cid: None for cid in self.client_data}
        last_plan: Dict[int, ClientPlan] = {}

        clients = sorted(self.client_data)
        for rnd in range(1, self.config.fl.num_rounds + 1):
            updates: List[Tuple[Params, int]] = []
            round_metrics: Dict[str, Dict] = {}

            for cid in clients:
                ctx = ClientContext(
                    client_id=cid,
                    client_type=self.client_types[cid],
                    predicted=self.cls[cid],
                    het_score=self.het[cid],
                )
                plan = self.spec.plan_fn(ctx)
                last_plan[cid] = plan
                ds = self.client_data[cid]["train"]

                if plan.kind == "single":
                    out, m = trainer.train_single_adapter(
                        self.config, ds, client_primary[cid], plan.epochs, self.gpu_id, cid,
                        freeze_a=plan.freeze_a,
                    )
                    client_primary[cid] = out
                    uploaded = out if plan.upload else None
                else:  # dual
                    g_out, l_out, m = trainer.train_dual_phase(
                        self.config, ds, client_primary[cid], client_local[cid],
                        plan.phases, self.gpu_id, cid,
                    )
                    client_primary[cid] = g_out
                    client_local[cid] = l_out
                    uploaded = g_out if plan.upload else None

                m.update({
                    "client_type": self.client_types[cid],
                    "predicted": self.cls[cid],
                    "het_score": self.het[cid],
                    "upload": plan.upload,
                    "inference": plan.inference,
                })
                if not plan.upload:
                    m["upload_bytes"] = 0
                round_metrics[str(cid)] = m
                if uploaded is not None:
                    updates.append((uploaded, m["num_examples"]))

            if updates:
                global_params = fedavg(updates)
                if self.spec.broadcast:
                    for cid in clients:
                        client_primary[cid] = global_params

            self.logger.log_round(rnd, {"client_metrics": round_metrics})
            self._print_round(rnd, round_metrics, n_uploads=len(updates))

        self._persist(global_params, client_primary, client_local, last_plan)

    def _print_round(self, rnd: int, round_metrics: Dict, n_uploads: int) -> None:
        losses = [m.get("train_loss", 0.0) for m in round_metrics.values()]
        mean_loss = sum(losses) / len(losses) if losses else 0.0
        print(f"[{self.spec.name} seed{self.config.fl.seed}] round {rnd:>2}/"
              f"{self.config.fl.num_rounds}  mean_train_loss={mean_loss:.4f}  "
              f"uploads={n_uploads}", flush=True)

    def _persist(
        self,
        global_params: Params,
        client_primary: Dict[int, Params],
        client_local: Dict[int, Optional[Params]],
        last_plan: Dict[int, ClientPlan],
    ) -> None:
        """Save adapters + manifest.json for the evaluator."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_params(global_params, self.run_dir / "global_adapter.npz")

        per_client_primary = not self.spec.broadcast  # only Local-Only keeps own φ_g
        clients_manifest: Dict[str, Dict] = {}
        for cid in sorted(self.client_data):
            plan = last_plan[cid]
            entry = {"type": self.client_types[cid], "inference": plan.inference}

            if per_client_primary:
                fname = f"client{cid}_primary.npz"
                trainer.save_params(client_primary[cid], self.run_dir / fname)
                entry["primary_file"] = fname
            else:
                entry["primary_file"] = "global_adapter.npz"

            if plan.inference == "global+local" and client_local[cid] is not None:
                fname = f"client{cid}_local.npz"
                trainer.save_params(client_local[cid], self.run_dir / fname)
                entry["local_file"] = fname
            else:
                entry["local_file"] = None

            clients_manifest[str(cid)] = entry

        manifest = {
            "method": self.spec.name,
            "seed": self.config.fl.seed,
            "experiment_name": self.config.experiment_name,
            "global_adapter": "global_adapter.npz",
            "clients": clients_manifest,
        }
        with open(self.run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved adapters + manifest to {self.run_dir}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one v2 (method, seed) experiment")
    parser.add_argument("--method", required=True, choices=sorted(REGISTRY))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index within the process (0 under CUDA_VISIBLE_DEVICES pinning)")
    args = parser.parse_args()

    config = build_config(args.method, args.seed)
    spec = get_method(args.method)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(LOGS_DIR / f"{config.experiment_name}.log")
    dump_config_yaml(config, CONFIGS_DIR / f"{config.experiment_name}.yaml")

    print(f"=== v2 run: {spec.name} (seed {args.seed}) ===")
    print(spec.description)

    het_scores, classification = get_scores_and_classification(config)
    server = V2Server(config, spec, het_scores, classification, gpu_id=args.gpu)
    server.run()


if __name__ == "__main__":
    main()
