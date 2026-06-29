"""Ablation studies for the v2 suite.

Three ablations, all reusing the *exact* main-v2 codebase, hyperparameters, and
GPT-4o-mini evaluation pipeline (no validation set, full 8-metric judge,
max_eval_samples=50, per-client IID/Non-IID breakdown). HA-DualLoRA is always the
1-epoch-TOTAL compute-matched variant (``ha_duallora_half``), so every method
trains exactly 1 local epoch per round.

  1. ratio_*    — vary the IID/Non-IID client *count* at fixed alpha (100 / 0.01)
  2. dataset_*  — generalize to EmoCareAI/Psych8k (data files swapped externally
                  by prepare_psych8k.py before this runs; restored after)
  3. ha_sched   — HA-DualLoRA adaptivity multiplier m; total epochs fixed at 1.0
                  (m=1 == ha_duallora_half canonical, m=0 == dual_lora_half)

Layout under ``experiments/v2/results/ablations/``::

    <setting>/<method>/        adapters, manifest.json, metrics.jsonl, het cache
    raw_scores/<setting>_<method>_client{cid}.jsonl
    summary/<setting>.json
    summary/ratio_comparison.json
    summary/_partial/<setting>_<method>_seed{seed}.json   (per-run eval result)

Configs are dumped to ``experiments/v2/configs/ablations/``. Nothing under the
main ``results/`` or ``configs/`` trees is read or written except, read-only, the
main ``results/summary/summary.json`` used as the Mixed midpoint in
``ratio_comparison``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from experiments.heterogeneity_aware_pfl.config import ExperimentConfig
from experiments.heterogeneity_aware_pfl.utils import setup_file_logging

from experiments.v2.evaluate_v2 import evaluate_run
from experiments.v2.het_score import get_scores_and_classification
from experiments.v2.methods import ClientPlan, ClientContext, MethodSpec, get_method
from experiments.v2.server_v2 import V2Server
from experiments.v2.v2_config import (
    CONFIGS_DIR,
    LOGS_DIR,
    RESULTS_DIR,
    build_config,
    dump_config_yaml,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ABLATIONS_DIR = RESULTS_DIR / "ablations"
ABL_RAW_DIR = ABLATIONS_DIR / "raw_scores"
ABL_SUMMARY_DIR = ABLATIONS_DIR / "summary"
ABL_PARTIAL_DIR = ABL_SUMMARY_DIR / "_partial"
ABL_CONFIGS_DIR = CONFIGS_DIR / "ablations"

# ---------------------------------------------------------------------------
# Setting definitions (all seed 42). ``methods`` are run in listed order.
# ---------------------------------------------------------------------------
STD_METHODS = ["fedavg", "ha_duallora_half", "selective"]

SETTINGS: Dict[str, Dict] = {
    # Ablation 1: IID/Non-IID ratio (count) sweep, MentalChat16K.
    "ratio_7iid_3noniid": {"num_iid": 7, "num_noniid": 3, "methods": STD_METHODS},
    "ratio_1iid_9noniid": {"num_iid": 1, "num_noniid": 9, "methods": STD_METHODS},
    # Ablation 2: dataset generalization to EmoCareAI/Psych8k (main 3/7 split).
    # Data files are swapped in by prepare_psych8k.py before this setting runs.
    "dataset_counselchat": {"num_iid": 3, "num_noniid": 7, "methods": STD_METHODS,
                            "dataset": "psych8k"},
    # Ablation 3: HA schedule adaptivity multiplier (main 3/7 split, MentalChat16K).
    # m=1 reproduces ha_duallora_half; m=0 reproduces dual_lora_half (both already
    # in main results), so only the off-center multipliers below are new runs.
    # Curve points: m = {0 (dual_half), 0.5, 1 (canonical), 1.5, 2}.
    "ha_sched": {"num_iid": 3, "num_noniid": 7,
                 "methods": ["ha_sched_m0.5", "ha_sched_m1.5", "ha_sched_m2.0"]},
}


# ---------------------------------------------------------------------------
# Method specs
# ---------------------------------------------------------------------------
def make_ha_sched_spec(m: float) -> MethodSpec:
    """HA-DualLoRA with a tunable adaptivity *multiplier* m, total epochs = 1.0.

    m scales the het->epoch-split steepness relative to the canonical
    ha_duallora_half (m=1), as a line in h with slope 0.5*m about the h=0.5 pivot:

        local epochs  = 0.5 + 0.5*m*(h - 0.5)
        global epochs = 0.5 - 0.5*m*(h - 0.5)

    m=0 == dual_lora_half (fixed 0.5/0.5, no adaptivity);
    m=1 == ha_duallora_half (canonical);
    m>1 == steeper. m is capped at 2.0: at m=2 the most concentrated clients (h=1)
    get global=0 and IID-like clients (h=0) get local=0 -- the steepest schedule
    before epochs would go negative. A 0-epoch phase is floored to a single step
    by the trainer, so neither adapter is ever fully skipped.
    """
    if not 0.0 <= m <= 2.0:
        raise ValueError(f"ha_sched multiplier m must be in [0, 2]; got {m}")

    def plan_fn(ctx: ClientContext) -> ClientPlan:
        h = min(1.0, max(0.0, ctx.het_score))
        delta = 0.5 * m * (h - 0.5)
        return ClientPlan(
            kind="dual",
            phases=[("local", 0.5 + delta), ("global", 0.5 - delta)],
            upload=True,
            inference="global+local",
        )

    return MethodSpec(
        name=f"ha_sched_m{m}",
        broadcast=True,
        uses_local=True,
        plan_fn=plan_fn,
        description=f"HA schedule adaptivity multiplier m={m} (x canonical); total epochs=1.0.",
    )


def get_spec(method: str) -> MethodSpec:
    """Resolve a method name to a MethodSpec (HA-schedule variants built locally)."""
    if method.startswith("ha_sched_m"):
        return make_ha_sched_spec(float(method[len("ha_sched_m"):]))
    return get_method(method)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def build_ablation_config(setting: str, method: str, seed: int = 42) -> ExperimentConfig:
    cfg = SETTINGS[setting]
    return build_config(
        method,
        seed,
        num_iid_clients=cfg["num_iid"],
        num_noniid_clients=cfg["num_noniid"],
        save_dir=str(ABLATIONS_DIR / setting),
        experiment_name=method,  # -> run dir results/ablations/<setting>/<method>/
    )


def _label(setting: str, method: str) -> str:
    """Globally-unique label for centralized raw-score / log filenames."""
    return f"{setting}_{method}"


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def train_one(setting: str, method: str, seed: int = 42, gpu: int = 0) -> None:
    config = build_ablation_config(setting, method, seed)
    spec = get_spec(method)

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ABL_CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    setup_file_logging(LOGS_DIR / f"abl_{_label(setting, method)}_seed{seed}.log")
    dump_config_yaml(config, ABL_CONFIGS_DIR / f"{setting}_{method}_seed{seed}.yaml")

    print(f"=== ablation train: {setting} / {spec.name} (seed {seed}) ===")
    print(spec.description)
    het_scores, classification = get_scores_and_classification(config)
    server = V2Server(config, spec, het_scores, classification, gpu_id=gpu)
    _exclude_empty_clients(server, setting, method)
    server.run()


def _exclude_empty_clients(server: V2Server, setting: str, method: str) -> None:
    """Drop clients with 0 training examples from the federation (and record it).

    Under an extreme split (e.g. 1 IID / 9 Non-IID at alpha=0.01) the Dirichlet
    draw can starve a client to 0 examples; it cannot be trained, aggregated, or
    evaluated. We exclude it (consistent across methods, since the partition is
    seed-fixed) and note the effective composition in the setting dir. The het
    scores already omit such clients, so server.het/cls stay consistent.
    """
    empty = sorted(cid for cid, cd in server.client_data.items() if len(cd["train"]) == 0)
    if not empty:
        return
    for cid in empty:
        server.client_data.pop(cid)
        server.client_types.pop(cid)
    remaining = sorted(server.client_data)
    note = {
        "excluded_clients": empty,
        "reason": "0 training examples (alpha=0.01 Dirichlet starvation)",
        "effective_clients": remaining,
        "effective_iid": [c for c in remaining if server.client_types[c] == "iid"],
        "effective_noniid": [c for c in remaining if server.client_types[c] == "noniid"],
    }
    setting_dir = ABLATIONS_DIR / setting
    setting_dir.mkdir(parents=True, exist_ok=True)
    with open(setting_dir / "excluded_clients.json", "w") as f:
        json.dump(note, f, indent=2)
    print(f"[ablation] {setting}/{method}: excluded empty client(s) {empty}; "
          f"effective = {len(note['effective_iid'])} IID + {len(note['effective_noniid'])} Non-IID")


def eval_one(setting: str, method: str, seed: int = 42, gpu: int = 0,
             force: bool = False) -> Optional[Dict]:
    config = build_ablation_config(setting, method, seed)
    result = evaluate_run(
        method, seed, gpu_id=gpu, force=force,
        config=config, raw_dir=ABL_RAW_DIR, exp_label=_label(setting, method),
    )
    if result is None:
        return None
    ABL_PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(ABL_PARTIAL_DIR / f"{setting}_{method}_seed{seed}.json", "w") as f:
        json.dump({"setting": setting, **result}, f, indent=2)
    return result


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def _groups_block(result: Dict) -> Dict:
    """Extract the reported quantities from an eval result."""
    g = result["groups"]
    return {
        "counseling": {grp: g.get(grp, {}).get("counseling") for grp in ("all", "iid", "noniid")},
        "personalization": {grp: g.get(grp, {}).get("personalization") for grp in ("all", "iid", "noniid")},
        "upload_mb": result["efficiency"].get("upload_mb"),
    }


def compile_summary(setting: str, seed: int = 42) -> Dict:
    """Aggregate the per-method eval partials for a setting into one summary JSON."""
    ABL_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    methods = SETTINGS[setting]["methods"]
    summary: Dict[str, Dict] = {}
    for method in methods:
        partial = ABL_PARTIAL_DIR / f"{setting}_{method}_seed{seed}.json"
        if not partial.exists():
            print(f"[summary] {setting}/{method}: no eval partial yet, skipping.")
            continue
        summary[method] = _groups_block(json.loads(partial.read_text()))
    out = {"setting": setting, "seed": seed, "methods": summary}
    excluded = ABLATIONS_DIR / setting / "excluded_clients.json"
    if excluded.exists():
        out["client_exclusion"] = json.loads(excluded.read_text())
    path = ABL_SUMMARY_DIR / f"{setting}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[summary] {setting} -> {path}")
    return out


# ---------------------------------------------------------------------------
# Ratio comparison (verify the expected personalization-gap trend)
# ---------------------------------------------------------------------------
MAIN_SUMMARY = RESULTS_DIR / "summary" / "summary.json"


def _main_personalization(key_base: str, group: str) -> Optional[float]:
    """Mean personalization for a main run across available seeds 42/99."""
    if not MAIN_SUMMARY.exists():
        return None
    main = json.loads(MAIN_SUMMARY.read_text())
    vals = []
    for s in (42, 99):
        entry = main.get(f"{key_base}_seed{s}")
        if entry and group in entry.get("groups", {}):
            v = entry["groups"][group].get("personalization")
            if v is not None:
                vals.append(v)
    return sum(vals) / len(vals) if vals else None


def _abl_personalization(setting: str, method: str, group: str, seed: int = 42) -> Optional[float]:
    path = ABL_SUMMARY_DIR / f"{setting}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    m = data.get("methods", {}).get(method)
    return m["personalization"].get(group) if m else None


def ratio_comparison(seed: int = 42) -> Dict:
    """Tabulate HA-DualLoRA(half) - FedAvg personalization gap across the ratio sweep.

    IID-heavy (7/3, ablation) -> Mixed (3/7, MAIN) -> Non-IID-heavy (1/9, ablation).
    The Mixed midpoint is read from the main summary (mean over seeds 42/99).
    """
    ABL_SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    points = [
        ("iid_heavy_7iid_3noniid", lambda g: (
            _abl_personalization("ratio_7iid_3noniid", "fedavg", g),
            _abl_personalization("ratio_7iid_3noniid", "ha_duallora_half", g))),
        ("mixed_3iid_7noniid_main", lambda g: (
            _main_personalization("fedavg", g),
            _main_personalization("ha_duallora_half", g))),
        ("noniid_heavy_1iid_9noniid", lambda g: (
            _abl_personalization("ratio_1iid_9noniid", "fedavg", g),
            _abl_personalization("ratio_1iid_9noniid", "ha_duallora_half", g))),
    ]
    table = []
    for name, getter in points:
        row = {"point": name}
        for group in ("all", "iid", "noniid"):
            fed, ha = getter(group)
            gap = (ha - fed) if (fed is not None and ha is not None) else None
            row[group] = {"fedavg": fed, "ha_duallora_half": ha, "ha_minus_fedavg": gap}
        table.append(row)
    out = {"seed": seed, "metric": "personalization (metric 8)", "points": table}
    path = ABL_SUMMARY_DIR / "ratio_comparison.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[ratio_comparison] -> {path}")
    # Convenience: print the 'all' gap trend the ablation is meant to verify.
    gaps = [r["all"]["ha_minus_fedavg"] for r in table]
    print(f"[ratio_comparison] HA-FedAvg personalization gap (all) "
          f"IID-heavy->Mixed->Non-IID-heavy: {gaps}")
    return out
