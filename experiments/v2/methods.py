"""Method registry for the v2 suite.

Each method is declarative: a :class:`MethodSpec` says how the server aggregates
and whether it broadcasts, plus a ``plan_fn`` that — given a client's id, ground-
truth type, Otsu-predicted class, and static het-score — returns a
:class:`ClientPlan` describing that client's local training for the round.

The server (`server_v2.py`) executes plans via the constant-LR trainer; it never
needs to know method-specific rules. Methods 6 (FDLoRA) and 7 (FFA-LoRA) are
deferred — see ``fusion.py`` / ``ffa_lora.py`` placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from experiments.v2.ffa_lora import plan_ffa_lora
from experiments.v2.fusion import plan_fdlora

# ---------------------------------------------------------------------------
# Per-client training plan
# ---------------------------------------------------------------------------
@dataclass
class ClientPlan:
    kind: str                                  # "single" | "dual"
    epochs: float = 1.0                        # single-adapter epochs
    phases: Optional[List[Tuple[str, float]]] = None  # dual: [(adapter, epochs), ...]
    upload: bool = True                        # contribute φ_g to FedAvg this round
    inference: str = "global"                  # "global" | "global+local"
    freeze_a: bool = False                     # single-adapter: freeze lora_A, train B only (FFA-LoRA)


@dataclass
class ClientContext:
    client_id: int
    client_type: str        # ground-truth "iid"/"noniid" (data partition)
    predicted: str          # Otsu-predicted "iid"/"noniid" (Method 5)
    het_score: float        # static entropy het-score in [0, 1]


@dataclass
class MethodSpec:
    name: str
    broadcast: bool                            # push aggregated φ_g to all clients
    uses_local: bool                           # any client keeps a local adapter
    plan_fn: Callable[[ClientContext], ClientPlan] = field(repr=False)
    description: str = ""


# ---------------------------------------------------------------------------
# Method 1: FedAvg (canon) — single global LoRA, all clients, FedAvg
# ---------------------------------------------------------------------------
def _plan_fedavg(ctx: ClientContext) -> ClientPlan:
    return ClientPlan(kind="single", epochs=1.0, upload=True, inference="global")


# Compute-matched control: FedAvg trained 2 epochs/round (same total local epochs
# as the dual-LoRA methods), to isolate architecture from training budget.
def _plan_fedavg_2ep(ctx: ClientContext) -> ClientPlan:
    return ClientPlan(kind="single", epochs=2.0, upload=True, inference="global")


# ---------------------------------------------------------------------------
# Method 2: Local-Only — independent per-client training, no aggregation
# ---------------------------------------------------------------------------
def _plan_local_only(ctx: ClientContext) -> ClientPlan:
    return ClientPlan(kind="single", epochs=1.0, upload=False, inference="global")


# ---------------------------------------------------------------------------
# Method 3: Dual LoRA (fixed) — local then global, 1 epoch each
# ---------------------------------------------------------------------------
def _plan_dual_lora(ctx: ClientContext) -> ClientPlan:
    return ClientPlan(
        kind="dual",
        phases=[("local", 1.0), ("global", 1.0)],
        upload=True,
        inference="global+local",
    )


# Compute-matched control: dual LoRA at a 1-epoch TOTAL budget (0.5 local + 0.5
# global), so it equals the 1-epoch single-adapter baselines instead of 2x.
def _plan_dual_lora_half(ctx: ClientContext) -> ClientPlan:
    return ClientPlan(
        kind="dual",
        phases=[("local", 0.5), ("global", 0.5)],
        upload=True,
        inference="global+local",
    )


# ---------------------------------------------------------------------------
# Method 4: HA-DualLoRA (proposed) — het-adaptive per-phase epochs
#   φ_l epochs = 0.5 + het   (more local specialization for concentrated clients)
#   φ_g epochs = 1.5 - het   (less global drift from concentrated clients)
# ---------------------------------------------------------------------------
def _plan_ha_duallora(ctx: ClientContext) -> ClientPlan:
    h = min(1.0, max(0.0, ctx.het_score))
    return ClientPlan(
        kind="dual",
        phases=[("local", 0.5 + h), ("global", 1.5 - h)],
        upload=True,
        inference="global+local",
    )


# ---------------------------------------------------------------------------
# Method 5: Selective Federated Aggregation — Otsu split
#   IID-like:     train φ_g 1 ep, upload φ_g
#   non-IID-like: freeze φ_g, train φ_l 1 ep, upload nothing
#   Server FedAvg over IID-like only; broadcast to all.
# ---------------------------------------------------------------------------
# Compute-matched control: HA-DualLoRA at a 1-epoch TOTAL budget. The canonical
# split (0.5+h local, 1.5-h global) sums to 2.0; halving keeps the het-driven
# ratio but normalizes the total to 1.0.
def _plan_ha_duallora_half(ctx: ClientContext) -> ClientPlan:
    h = min(1.0, max(0.0, ctx.het_score))
    return ClientPlan(
        kind="dual",
        phases=[("local", (0.5 + h) / 2.0), ("global", (1.5 - h) / 2.0)],
        upload=True,
        inference="global+local",
    )


def _plan_selective(ctx: ClientContext) -> ClientPlan:
    if ctx.predicted == "iid":
        return ClientPlan(kind="single", epochs=1.0, upload=True, inference="global")
    # non-IID-like: φ_g absent from phases => frozen and not uploaded.
    return ClientPlan(
        kind="dual",
        phases=[("local", 1.0)],
        upload=False,
        inference="global+local",
    )


REGISTRY = {
    "fedavg": MethodSpec(
        name="fedavg", broadcast=True, uses_local=False, plan_fn=_plan_fedavg,
        description="FedAvg (canon): single global LoRA r16, all clients, FedAvg.",
    ),
    "fedavg_2ep": MethodSpec(
        name="fedavg_2ep", broadcast=True, uses_local=False, plan_fn=_plan_fedavg_2ep,
        description="FedAvg, 2 epochs/round: compute-matched control vs dual-LoRA.",
    ),
    "local_only": MethodSpec(
        name="local_only", broadcast=False, uses_local=False, plan_fn=_plan_local_only,
        description="Local-Only: independent per-client training, no aggregation.",
    ),
    "dual_lora": MethodSpec(
        name="dual_lora", broadcast=True, uses_local=True, plan_fn=_plan_dual_lora,
        description="Dual LoRA (fixed): local then global, 1 epoch each; upload φ_g.",
    ),
    "ha_duallora": MethodSpec(
        name="ha_duallora", broadcast=True, uses_local=True, plan_fn=_plan_ha_duallora,
        description="HA-DualLoRA: het-adaptive per-phase epochs; upload φ_g.",
    ),
    "dual_lora_half": MethodSpec(
        name="dual_lora_half", broadcast=True, uses_local=True, plan_fn=_plan_dual_lora_half,
        description="Dual LoRA at 1-epoch budget (0.5 local + 0.5 global): compute-matched.",
    ),
    "ha_duallora_half": MethodSpec(
        name="ha_duallora_half", broadcast=True, uses_local=True, plan_fn=_plan_ha_duallora_half,
        description="HA-DualLoRA at 1-epoch budget (het ratio, total=1.0): compute-matched.",
    ),
    "selective": MethodSpec(
        name="selective", broadcast=True, uses_local=True, plan_fn=_plan_selective,
        description="Selective: IID-like upload φ_g; non-IID-like train φ_l only.",
    ),
    # --- Deferred (registered so they are discoverable, but plan_fn raises) ---
    "fdlora": MethodSpec(
        name="fdlora", broadcast=True, uses_local=True, plan_fn=plan_fdlora,
        description="FDLoRA (Method 6) — DEFERRED.",
    ),
    "ffa_lora": MethodSpec(
        name="ffa_lora", broadcast=True, uses_local=False, plan_fn=plan_ffa_lora,
        description="FFA-LoRA (Method 7) — DEFERRED.",
    ),
}


def get_method(name: str) -> MethodSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Available: {sorted(REGISTRY)}")
    return REGISTRY[name]
