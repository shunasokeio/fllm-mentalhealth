"""Method 6 — FDLoRA (arXiv:2406.07925).

FDLoRA keeps two LoRA adapters per client — a *personalized* adapter (PEFT name
``local``, never communicated) and a *global* adapter (``default``, communicated
via a Nesterov outer optimizer) — and runs in three stages:

  Stage 1  Local SFT of the personalized adapter; the server then initializes the
           global adapter as the simple mean of all personalized adapters.
  Stage 2  Federated rounds: clients train the global adapter (personalized
           frozen but present in the forward), the server applies a Nesterov
           outer step, and every ``H`` rounds the global is copied back into each
           client's personalized adapter. Held-out loss is logged every 2 rounds.
  Stage 3  AdaFusion: each client finds blend weights ``(w1, w2)`` that minimize
           held-out cross-entropy of the cross-blended adapter
           ``(w1*B_p + w2*B_g) @ (w1*A_p + w2*A_g) * scale`` + L1(w), via a
           gradient-free Nelder-Mead search.

This cannot be expressed as a per-round ``plan_fn`` (it needs a server-side outer
optimizer, a pre-stage, and a post-stage fusion), so FDLoRA has its own runner and
entry point:

    CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.fusion --seed 42

It reuses the canonical v2 config (``build_config``) and the constant-LR trainer
primitives, so its LR / rank / rounds / seeds match every other v2 method.

Canonical-HP overrides vs the FDLoRA paper (warned at startup): LR 2e-4 -> 2e-5,
rounds 30 -> 10. The outer-optimizer constants (LR_OUTER, OUTER_MOMENTUM) belong
to the method architecture and are kept at the paper values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from scipy.optimize import minimize
from torch.utils.data import DataLoader

from experiments.heterogeneity_aware_pfl.config import ExperimentConfig
from experiments.heterogeneity_aware_pfl.data_utils import prepare_all_clients
from experiments.heterogeneity_aware_pfl.model_utils import (
    activate_adapters,
    get_model,
    get_tokenizer_and_data_collator,
    set_adapter_params,
)
from experiments.heterogeneity_aware_pfl.utils import (
    ExperimentLogger,
    set_seed,
    setup_file_logging,
)

from experiments.v2 import trainer
from experiments.v2.v2_config import CONFIGS_DIR, LOGS_DIR, build_config, dump_config_yaml

Params = List[np.ndarray]

# ---------------------------------------------------------------------------
# FDLoRA constants
# ---------------------------------------------------------------------------
# Outer-optimizer architecture (paper values; NOT part of the canonical HP set).
LR_OUTER = 0.001
OUTER_MOMENTUM = 0.5
SYNC_EVERY = 1          # H: copy global -> personalized every H rounds
EVAL_EVERY = 2          # log held-out loss every EVAL_EVERY rounds
VAL_FRACTION = 0.02     # local held-out split for AdaFusion + mid-train eval
# AdaFusion (gradient-free blend search)
ADAFUSION_X0 = [0.5, 0.5]
ADAFUSION_MAXITER = 5
ADAFUSION_L1 = 0.05
# Paper defaults we override with canonical v2 values (for the startup warning).
PAPER_LR = 2e-4
PAPER_ROUNDS = 30

_LOSS_BATCH_SIZE = 8


# ---------------------------------------------------------------------------
# Method-registry stub: FDLoRA does not run through the generic server.
# ---------------------------------------------------------------------------
def plan_fdlora(ctx):  # noqa: ANN001 - matches methods.ClientContext
    raise NotImplementedError(
        "FDLoRA (Method 6) has a dedicated runner (server-side outer optimizer + "
        "AdaFusion). Run it with: python -m experiments.v2.fusion --seed <seed>"
    )


# ---------------------------------------------------------------------------
# Building block 1: Nesterov outer optimizer (operates on Params)
# ---------------------------------------------------------------------------
def nesterov_outer_step(
    prev_global: Params, client_globals: List[Params], velocity: Params,
) -> Tuple[Params, Params]:
    """One Nesterov-SGD outer step over the clients' returned global adapters.

    delta    = mean_c (prev_global - client_global_c)        (unweighted)
    velocity = OUTER_MOMENTUM * velocity + delta             (persists across rounds)
    global   = prev_global - LR_OUTER * velocity
    """
    new_velocity: Params = []
    new_global: Params = []
    for i, pg in enumerate(prev_global):
        delta = np.mean([pg - cg[i] for cg in client_globals], axis=0)
        v = OUTER_MOMENTUM * velocity[i] + delta
        new_velocity.append(v)
        new_global.append(pg - LR_OUTER * v)
    return new_global, new_velocity


# ---------------------------------------------------------------------------
# Building block 2: cross-blended fused delta + forward hooks
# ---------------------------------------------------------------------------
def _fused_delta(x, A_p, B_p, A_g, B_g, scale: float, w1: float, w2: float):
    """Delta a fused adapter adds to a linear layer's output for input ``x``.

    Equivalent to (w1*B_p + w2*B_g) @ (w1*A_p + w2*A_g) applied to x, times scale.
    A_* are [r, in], B_* are [out, r]; x is [..., in]; returns [..., out].
    """
    fused_A = w1 * A_p + w2 * A_g
    fused_B = w1 * B_p + w2 * B_g
    return scale * (x @ fused_A.t()) @ fused_B.t()


class FusedHookManager:
    """Apply the cross-blended fused adapter via forward hooks on a 2-adapter model.

    PEFT sums adapters, so it cannot produce the w1*w2 cross terms in
    ``(w1*B_p + w2*B_g) @ (w1*A_p + w2*A_g)``. Instead we disable PEFT's own
    adapter contribution and add the fused delta to each LoRA layer's output in a
    forward hook. ``set_weights`` lets the scipy search update (w1, w2) in place
    without re-registering hooks.
    """

    def __init__(self, model, w1: float = 0.5, w2: float = 0.5):
        self.model = model
        self.w = [float(w1), float(w2)]
        self.handles = []
        self._layers = []  # (A_p, B_p, A_g, B_g, scale) tensor refs per LoRA layer
        for module in model.modules():
            if not (hasattr(module, "lora_A") and "default" in getattr(module, "lora_A", {})):
                continue
            if "local" not in module.lora_A:
                continue
            refs = (
                module.lora_A["local"].weight,   # A_p
                module.lora_B["local"].weight,   # B_p
                module.lora_A["default"].weight,  # A_g
                module.lora_B["default"].weight,  # B_g
                float(module.scaling["default"]),
            )
            self._layers.append(refs)
            self.handles.append(module.register_forward_hook(self._make_hook(refs)))
        if not self._layers:
            raise RuntimeError("FDLoRA fusion: no LoRA layers with both adapters found")
        # Disable PEFT's own adapter math so the forward returns base output only;
        # our hooks then add exactly the fused delta.
        self.model.base_model.disable_adapter_layers()

    def _make_hook(self, refs):
        A_p, B_p, A_g, B_g, scale = refs

        def hook(_module, inputs, output):
            w1, w2 = self.w
            delta = _fused_delta(inputs[0], A_p, B_p, A_g, B_g, scale, w1, w2)
            return output + delta.to(output.dtype)

        return hook

    def set_weights(self, w1: float, w2: float) -> None:
        self.w = [float(w1), float(w2)]

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        self.model.base_model.enable_adapter_layers()


# ---------------------------------------------------------------------------
# Building block 3: held-out cross-entropy + AdaFusion search
# ---------------------------------------------------------------------------
def _dataset_loss(model, collator, dataset, device, batch_size: int = _LOSS_BATCH_SIZE) -> float:
    """Token-weighted mean cross-entropy of ``model`` over ``dataset``."""
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    total_loss, total_tok = 0.0, 0
    model.eval()
    use_autocast = device == "cuda" and torch.cuda.is_available()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                out = model(**batch)
            ntok = int((batch["labels"] != -100).sum().item())
            if ntok == 0:
                continue
            total_loss += float(out.loss.item()) * ntok
            total_tok += ntok
    return total_loss / max(1, total_tok)


def adafusion(model, collator, val_ds, device) -> Tuple[float, float]:
    """Nelder-Mead search for the best blend weights (w1, w2) on a held-out split.

    Objective = held-out CE of the fused model + L1 regularization
    ``ADAFUSION_L1 * (|w1| + |w2|)``. Returns (w1, w2).
    """
    manager = FusedHookManager(model, *ADAFUSION_X0)

    def objective(w):
        manager.set_weights(float(w[0]), float(w[1]))
        ce = _dataset_loss(model, collator, val_ds, device)
        return ce + ADAFUSION_L1 * (abs(float(w[0])) + abs(float(w[1])))

    res = minimize(objective, np.array(ADAFUSION_X0, dtype=float),
                   method="Nelder-Mead", options={"maxiter": ADAFUSION_MAXITER})
    manager.remove()
    return float(res.x[0]), float(res.x[1])


# ---------------------------------------------------------------------------
# Param helpers
# ---------------------------------------------------------------------------
def _mean_params(param_lists: List[Params]) -> Params:
    """Element-wise simple mean of a list of equal-shaped Params."""
    n = len(param_lists)
    return [np.mean([pl[i] for pl in param_lists], axis=0) for i in range(len(param_lists[0]))]


# ---------------------------------------------------------------------------
# FDLoRA server / runner
# ---------------------------------------------------------------------------
class FDLoRAServer:
    def __init__(self, config: ExperimentConfig, gpu_id: int = 0):
        self.config = config
        self.gpu_id = gpu_id
        self.client_data = prepare_all_clients(config)
        self.client_types = {cid: cd["type"] for cid, cd in self.client_data.items()}
        self.logger = ExperimentLogger(config.save_dir, config.experiment_name)
        self.run_dir = Path(config.save_dir) / config.experiment_name
        self.tokenizer, self.collator = get_tokenizer_and_data_collator(config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Carve a deterministic 2% held-out split from each client's train set.
        # The 98% remainder is used for ALL training (Stages 1 & 2); the 2% is
        # used only for AdaFusion and mid-training held-out loss.
        self.train: Dict[int, object] = {}
        self.val: Dict[int, object] = {}
        for cid, cd in self.client_data.items():
            split = cd["train"].train_test_split(
                test_size=VAL_FRACTION, seed=config.fl.seed + cid)
            self.train[cid] = split["train"]
            self.val[cid] = split["test"]

    # -- Stage 1 ----------------------------------------------------------------
    def _stage1_local_sft(self, init_global: Params) -> Dict[int, Params]:
        personalized: Dict[int, Params] = {}
        for cid in sorted(self.client_data):
            _g, l_out, m = trainer.train_dual_phase(
                self.config, self.train[cid], init_global, None,
                [("local", 1.0)], self.gpu_id, cid,
            )
            personalized[cid] = l_out
            self._record(1, cid, m, phase="stage1_local_sft")
            print(f"[fdlora seed{self.config.fl.seed}] stage1 client {cid} "
                  f"local_loss={m.get('train_loss_local'):.4f}", flush=True)
        return personalized

    # -- Stage 2 ----------------------------------------------------------------
    def _stage2_federated(
        self, global_params: Params, personalized: Dict[int, Params],
    ) -> Params:
        clients = sorted(self.client_data)
        velocity: Params = [np.zeros_like(p) for p in global_params]
        for rnd in range(1, self.config.fl.num_rounds + 1):
            prev_global = global_params
            client_globals: List[Params] = []
            round_metrics: Dict[str, Dict] = {}
            for cid in clients:
                g_out, _l_out, m = trainer.train_dual_phase(
                    self.config, self.train[cid], prev_global, personalized[cid],
                    [("global", 1.0)], self.gpu_id, cid,
                )
                client_globals.append(g_out)
                m.update({"client_type": self.client_types[cid], "stage": "stage2_federated"})
                round_metrics[str(cid)] = m

            global_params, velocity = nesterov_outer_step(prev_global, client_globals, velocity)

            # H=SYNC_EVERY: copy the current global into every personalized adapter.
            if rnd % SYNC_EVERY == 0:
                for cid in clients:
                    personalized[cid] = [p.copy() for p in global_params]

            # Mid-training eval: held-out CE of the global LoRA (no fusion).
            if rnd % EVAL_EVERY == 0:
                losses = self._heldout_losses(global_params)
                for cid in clients:
                    round_metrics[str(cid)]["heldout_loss"] = losses[cid]
                mean_hl = float(np.mean(list(losses.values())))
                print(f"[fdlora seed{self.config.fl.seed}] round {rnd:>2}/"
                      f"{self.config.fl.num_rounds} mean_heldout_loss={mean_hl:.4f}", flush=True)

            self.logger.log_round(rnd, {"client_metrics": round_metrics})
        return global_params

    def _heldout_losses(self, global_params: Params) -> Dict[int, float]:
        """Held-out CE of the global adapter on each client's 2% val split."""
        model = get_model(self.config, gpu_id=self.gpu_id)
        set_adapter_params(model, global_params, adapter_name="default")
        activate_adapters(model, "default")
        losses = {cid: _dataset_loss(model, self.collator, self.val[cid], self.device)
                  for cid in sorted(self.client_data)}
        _free(model)
        return losses

    # -- Stage 3 ----------------------------------------------------------------
    def _stage3_adafusion(
        self, global_params: Params, personalized: Dict[int, Params],
    ) -> Dict[int, Tuple[float, float]]:
        fusion_weights: Dict[int, Tuple[float, float]] = {}
        for cid in sorted(self.client_data):
            model = _build_two_adapter_model(
                self.config, global_params, personalized[cid], self.gpu_id)
            w1, w2 = adafusion(model, self.collator, self.val[cid], self.device)
            fusion_weights[cid] = (w1, w2)
            _free(model)
            print(f"[fdlora seed{self.config.fl.seed}] stage3 client {cid} "
                  f"w1={w1:.3f} w2={w2:.3f}", flush=True)
        return fusion_weights

    # -- Orchestration ----------------------------------------------------------
    def run(self) -> None:
        set_seed(self.config.fl.seed)
        self.logger.reset_round_logs()
        _warn_canonical_overrides(self.config)

        init_global = trainer.initial_global_params(self.config, gpu_id=self.gpu_id)
        print(f"[fdlora seed{self.config.fl.seed}] Stage 1: local SFT", flush=True)
        personalized = self._stage1_local_sft(init_global)

        global_params = _mean_params([personalized[c] for c in sorted(personalized)])
        print(f"[fdlora seed{self.config.fl.seed}] Stage 2: {self.config.fl.num_rounds} "
              f"federated rounds (Nesterov outer optimizer)", flush=True)
        global_params = self._stage2_federated(global_params, personalized)

        print(f"[fdlora seed{self.config.fl.seed}] Stage 3: AdaFusion", flush=True)
        fusion_weights = self._stage3_adafusion(global_params, personalized)

        self._persist(global_params, personalized, fusion_weights)

    def _record(self, rnd: int, cid: int, m: Dict, phase: str) -> None:
        m = dict(m)
        m.update({"client_type": self.client_types[cid], "stage": phase})
        self.logger.log_round(rnd, {"client_metrics": {str(cid): m}})

    def _persist(
        self, global_params: Params, personalized: Dict[int, Params],
        fusion_weights: Dict[int, Tuple[float, float]],
    ) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_params(global_params, self.run_dir / "global_adapter.npz")
        clients_manifest: Dict[str, Dict] = {}
        for cid in sorted(self.client_data):
            lname = f"client{cid}_local.npz"
            trainer.save_params(personalized[cid], self.run_dir / lname)
            w1, w2 = fusion_weights[cid]
            clients_manifest[str(cid)] = {
                "type": self.client_types[cid],
                "inference": "fused",
                "primary_file": "global_adapter.npz",
                "local_file": lname,
                "w1": w1,
                "w2": w2,
            }
        manifest = {
            "method": "fdlora",
            "seed": self.config.fl.seed,
            "experiment_name": self.config.experiment_name,
            "global_adapter": "global_adapter.npz",
            "clients": clients_manifest,
        }
        with open(self.run_dir / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved adapters + manifest to {self.run_dir}", flush=True)


# ---------------------------------------------------------------------------
# Model construction (two adapters: default=global, local=personalized)
# ---------------------------------------------------------------------------
def _build_two_adapter_model(config: ExperimentConfig, global_params: Params,
                             personalized: Params, gpu_id: int):
    model = get_model(config, gpu_id=gpu_id)
    set_adapter_params(model, global_params, adapter_name="default")
    trainer.add_local_adapter_v2(model, config)
    set_adapter_params(model, personalized, adapter_name="local")
    activate_adapters(model, ["default", "local"])
    return model


def build_fused_eval_model(config: ExperimentConfig, run_dir: Path, entry: Dict, gpu_id: int = 0):
    """Reconstruct a client's fused model for evaluation (hooks stay active)."""
    global_params = trainer.load_params(run_dir / entry["primary_file"])
    personalized = trainer.load_params(run_dir / entry["local_file"])
    model = _build_two_adapter_model(config, global_params, personalized, gpu_id)
    FusedHookManager(model, float(entry["w1"]), float(entry["w2"]))  # hooks persist for generation
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
def _warn_canonical_overrides(config: ExperimentConfig) -> None:
    if abs(config.train.lr_max - PAPER_LR) > 1e-12:
        print(f"WARNING: FDLoRA LR overridden to canonical {config.train.lr_max:g} "
              f"(paper default {PAPER_LR:g})", flush=True)
    if config.fl.num_rounds != PAPER_ROUNDS:
        print(f"WARNING: FDLoRA NUM_ROUNDS overridden to canonical {config.fl.num_rounds} "
              f"(paper default {PAPER_ROUNDS})", flush=True)


def _free(model) -> None:
    import gc
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Isolated unit tests (build-block validation, no GPU/model needed)
# ---------------------------------------------------------------------------
def _self_test() -> None:
    # 1) Nesterov outer step, two dummy clients, hand-checked.
    prev = [np.array([1.0, 2.0]), np.array([[0.0, 0.0]])]
    c1 = [np.array([0.0, 2.0]), np.array([[1.0, 1.0]])]
    c2 = [np.array([2.0, 0.0]), np.array([[-1.0, 1.0]])]
    vel = [np.zeros_like(p) for p in prev]
    new_g, new_v = nesterov_outer_step(prev, [c1, c2], vel)
    # delta_0 = mean([1,0],[-1,2]) = [0,1]; v=0.5*0+delta; g=prev-0.001*v
    assert np.allclose(new_v[0], [0.0, 1.0]), new_v[0]
    assert np.allclose(new_g[0], [1.0, 2.0 - 0.001 * 1.0]), new_g[0]
    # delta_1 = mean([-1,-1],[1,-1]) = [0,-1]
    assert np.allclose(new_v[1], [[0.0, -1.0]]), new_v[1]
    assert np.allclose(new_g[1], [[0.0, 0.001]]), new_g[1]

    # 2) Fused delta vs an independent manual computation.
    torch.manual_seed(0)
    r, din, dout = 4, 6, 5
    A_p, A_g = torch.randn(r, din), torch.randn(r, din)
    B_p, B_g = torch.randn(dout, r), torch.randn(dout, r)
    x = torch.randn(3, din)
    scale = 0.5
    for w1, w2 in [(0.5, 0.5), (1.0, 0.0), (0.3, 0.7)]:
        got = _fused_delta(x, A_p, B_p, A_g, B_g, scale, w1, w2)
        fused_A = w1 * A_p + w2 * A_g
        fused_B = w1 * B_p + w2 * B_g
        manual = scale * (fused_B @ (fused_A @ x.t())).t()  # per-sample B@(A@x)
        assert torch.allclose(got, manual, atol=1e-5), (w1, w2)

    print("fusion._self_test: OK (nesterov_outer_step + _fused_delta)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one FDLoRA (seed) experiment")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--self-test", action="store_true",
                        help="run isolated build-block unit tests and exit")
    args = parser.parse_args()

    if args.self_test:
        _self_test()
        return
    if args.seed is None:
        parser.error("--seed is required (unless --self-test)")

    config = build_config("fdlora", args.seed)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(LOGS_DIR / f"{config.experiment_name}.log")
    dump_config_yaml(config, CONFIGS_DIR / f"{config.experiment_name}.yaml")

    print(f"=== v2 run: fdlora (seed {args.seed}) ===")
    server = FDLoRAServer(config, gpu_id=args.gpu)
    server.run()


if __name__ == "__main__":
    main()
