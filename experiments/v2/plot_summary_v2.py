"""Comparison plots for the v2 suite: counseling, personalization, and the
quality/cost frontier, with the untrained base model as reference.

Reads results/summary/summary.json (per method x seed) and
results/base_seed*/base_results.json, averages over seeds, writes PNGs to
results/summary/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from experiments.v2.v2_config import RESULTS_DIR, V2_SEEDS

SUMMARY = RESULTS_DIR / "summary"
METHODS = ["fedavg", "ffa_lora", "local_only", "selective",
           "dual_lora_half", "ha_duallora_half", "dual_lora", "ha_duallora"]
GROUPS = ["all", "iid", "noniid"]


def seed_avg(summary, method, group, key):
    vals = [summary[f"{method}_seed{s}"]["groups"][group][key]
            for s in V2_SEEDS if f"{method}_seed{s}" in summary]
    return float(np.mean(vals))


def eff_avg(summary, method, key):
    vals = [summary[f"{method}_seed{s}"]["efficiency"][key]
            for s in V2_SEEDS if f"{method}_seed{s}" in summary]
    return float(np.mean([v for v in vals if v is not None]))


def base_avg(group, key):
    vals = []
    for s in V2_SEEDS:
        p = RESULTS_DIR / f"base_seed{s}" / "base_results.json"
        if p.exists():
            vals.append(json.load(open(p))["groups"][group][key])
    return float(np.mean(vals)) if vals else None


def grouped_bar(summary, metric, title, fname):
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(METHODS))
    w = 0.26
    colors = {"all": "#4C72B0", "iid": "#55A868", "noniid": "#C44E52"}
    for i, g in enumerate(GROUPS):
        vals = [seed_avg(summary, m, g, metric) for m in METHODS]
        bars = ax.bar(x + (i - 1) * w, vals, w, label=g, color=colors[g])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7)
        b = base_avg(g, metric)
        if b is not None:
            ax.axhline(b, color=colors[g], ls="--", lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=15)
    ax.set_ylabel(metric.capitalize() + " score (1-10)")
    ax.set_title(f"{title}\n(dashed = untrained base per group; seed-averaged)")
    ax.legend(title="client group"); ax.grid(axis="y", alpha=0.3)
    lo = min(base_avg(g, metric) for g in GROUPS) - 0.2
    hi = max(seed_avg(summary, m, "noniid", metric) for m in METHODS) + 0.25
    ax.set_ylim(lo, hi)
    fig.tight_layout(); fig.savefig(SUMMARY / fname, dpi=140); plt.close(fig)
    print("wrote", SUMMARY / fname)


def frontier(summary, xkey, xlabel, title, fname, annotate_key=None):
    """Quality (personalization) vs a cost axis (xkey), seed-averaged."""
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for m in METHODS:
        x = eff_avg(summary, m, xkey)
        y = seed_avg(summary, m, "all", "personalization")
        ax.scatter(x, y, s=120)
        label = m
        if annotate_key:
            label = f"{m}\n({eff_avg(summary, m, annotate_key):.0f} {annotate_key.split('_')[0]})"
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 4), fontsize=9)
    yb = base_avg("all", "personalization")
    if yb is not None:
        ax.axhline(yb, color="gray", ls="--", lw=1, label=f"base ({yb:.2f})")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Personalization (all, 1-10)")
    ax.set_title(f"{title} (seed-averaged)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(SUMMARY / fname, dpi=140); plt.close(fig)
    print("wrote", SUMMARY / fname)


# Per-method efficiency metrics: (summary key, axis label, bar color).
EFF_PANELS = [
    ("upload_mb", "Upload (MB / client / round)", "#8172B3"),
    ("compute_time_s", "Compute time (s / client / round)", "#CCB974"),
    ("peak_gpu_memory_mb", "Peak GPU memory (MB)", "#64B5CD"),
]


def efficiency_bars(summary, fname):
    """Per-method communication / compute / memory cost (FFA-LoRA's B-only upload
    shows here as ~half of fedavg)."""
    fig, axes = plt.subplots(1, len(EFF_PANELS), figsize=(15, 5))
    x = np.arange(len(METHODS))
    for ax, (key, label, color) in zip(axes, EFF_PANELS):
        vals = [eff_avg(summary, m, key) for m in METHODS]
        bars = ax.bar(x, vals, color=color)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(METHODS, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-client/round efficiency by method (seed-averaged)")
    fig.tight_layout(); fig.savefig(SUMMARY / fname, dpi=140); plt.close(fig)
    print("wrote", SUMMARY / fname)


def main():
    summary = json.load(open(SUMMARY / "summary.json"))
    grouped_bar(summary, "counseling", "Counseling Quality (7-metric)", "plot_counseling.png")
    grouped_bar(summary, "personalization", "Personalization (standalone)", "plot_personalization.png")
    frontier(summary, "compute_time_s", "Compute time (s / client / round)",
             "Quality vs compute frontier", "plot_frontier.png", annotate_key="upload_mb")
    frontier(summary, "upload_mb", "Upload (MB / client / round)",
             "Quality vs communication frontier", "plot_frontier_upload.png")
    efficiency_bars(summary, "plot_efficiency.png")


if __name__ == "__main__":
    main()
