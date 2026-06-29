"""Method 7 — FFA-LoRA (arXiv:2403.12313).

FFA-LoRA is plain FedAvg over a single LoRA (rank 16) with one change: matrix A
is *frozen* at its PEFT Gaussian initialization and only matrix B is trained and
communicated. Since A is broadcast identically to every client (the server's
round-1 init) and never updated, full-adapter FedAvg is numerically identical to
B-only FedAvg, so the canonical v2 server/aggregation/broadcast path is reused
unchanged — only training freezes A (see ``trainer.train_single_adapter`` with
``freeze_a=True``), and the reported ``upload_bytes`` count B alone.

Canonical HP: rank 16, alpha 8 (scaling 0.5, matching every v2 method), LR 2e-5,
1 local epoch, FedAvg, broadcast. No local adapter.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # avoid a circular import at runtime (methods imports this module)
    from experiments.v2.methods import ClientContext, ClientPlan


def plan_ffa_lora(ctx: "ClientContext") -> "ClientPlan":
    """Single global LoRA, all clients, FedAvg — but with lora_A frozen (B only)."""
    from experiments.v2.methods import ClientPlan

    return ClientPlan(
        kind="single",
        epochs=1.0,
        upload=True,
        inference="global",
        freeze_a=True,
    )
