"""Counseling-quality summary figure, epoch-matched.

Same grouped-bar style as plot_summary_v2.py's counseling panel, but the
dual-adapter methods are represented *only* by their compute-matched (~1 epoch
/ client / round) variants: dual_lora_half and ha_duallora_half. The full-budget
dual_lora / ha_duallora arms (which train two adapters = ~2 epochs of compute)
are excluded so every method on the figure is compared at the same local budget.

Reads results/summary/summary.json and results/base_seed*/base_results.json,
averages over seeds, writes results/summary/plot_counseling_epoch_matched.png.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.v2.v2_config import RESULTS_DIR, V2_SEEDS

SUMMARY = RESULTS_DIR / "summary"

# Epoch-matched method set: single-adapter methods + the _half dual variants.
METHODS = ["fedavg", "ffa_lora", "local_only", "selective",
           "dual_lora_half", "ha_duallora_half"]
LABELS = {
    "fedavg": "FedAvg",
    "ffa_lora": "FFA-LoRA",
    "local_only": "Local-only",
    "selective": "Selective",
    "dual_lora_half": "DualLoRA\n(epoch-matched)",
    "ha_duallora_half": "HA-DualLoRA\n(epoch-matched)",
}
GROUPS = ["all", "iid", "noniid"]
METRIC = "counseling"


def seed_avg(summary, method, group, key):
    vals = [summary[f"{method}_seed{s}"]["groups"][group][key]
            for s in V2_SEEDS if f"{method}_seed{s}" in summary]
    return float(np.mean(vals))


def base_avg(group, key):
    vals = []
    for s in V2_SEEDS:
        p = RESULTS_DIR / f"base_seed{s}" / "base_results.json"
        if p.exists():
            vals.append(json.load(open(p))["groups"][group][key])
    return float(np.mean(vals)) if vals else None


def main():
    summary = json.load(open(SUMMARY / "summary.json"))
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(METHODS))
    w = 0.26
    colors = {"all": "#4C72B0", "iid": "#55A868", "noniid": "#C44E52"}
    for i, g in enumerate(GROUPS):
        vals = [seed_avg(summary, m, g, METRIC) for m in METHODS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=g, color=colors[g])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
        b = base_avg(g, METRIC)
        if b is not None:
            ax.axhline(b, color=colors[g], ls="--", lw=1, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[m] for m in METHODS], rotation=15)
    ax.set_ylabel("Counseling score (1-10)")
    ax.set_title("Counseling Quality (7-metric), epoch-matched\n"
                 "(dual variants at ~1 epoch budget; dashed = untrained base per group; seed-averaged)")
    ax.legend(title="client group")
    ax.grid(axis="y", alpha=0.3)
    bases = [base_avg(g, METRIC) for g in GROUPS]
    lo = min(b for b in bases if b is not None) - 0.2
    hi = max(seed_avg(summary, m, g, METRIC)
             for m in METHODS for g in GROUPS) + 0.25
    ax.set_ylim(lo, hi)
    fig.tight_layout()
    out = SUMMARY / "plot_counseling_epoch_matched.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
