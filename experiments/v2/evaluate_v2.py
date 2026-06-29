"""v2 evaluation & reporting.

Runs the GPT-4o-mini 8-metric judge (temperature 0) on the FULL test set for each
trained run, then reports — per method x seed x group (all / IID / non-IID):

  * Counseling Quality  = mean of the 7 counseling metrics (metrics 1-7)
  * Personalization     = the 8th metric ("Personalization & Contextual
                          Adaptation") reported standalone (NOT folded in)
  * Efficiency          = upload MB, compute time (s), peak GPU MB per client/round

Per-sample judge scores are written to ``results/raw_scores/``; summary tables to
``results/summary/`` (JSON + CSV); per-client scores to ``logs/``.

    CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2
    CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 --methods fedavg --seeds 42
"""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from experiments.heterogeneity_aware_pfl.config import ExperimentConfig
from experiments.heterogeneity_aware_pfl.data_utils import prepare_all_clients
import torch
from tqdm import tqdm

from experiments.heterogeneity_aware_pfl.evaluate import (
    COMPOSITE_METRICS,
    gpt_evaluate_single,
)
from experiments.heterogeneity_aware_pfl.model_utils import (
    activate_adapters,
    get_model,
    get_tokenizer_and_data_collator,
    set_adapter_params,
)

from experiments.v2 import trainer
from experiments.v2.v2_config import (
    LOGS_DIR,
    RESULTS_DIR,
    V2_METHODS,
    V2_SEEDS,
    build_config,
)

EVAL_BATCH_SIZE = 16
JUDGE_WORKERS = 8  # concurrent GPT-4o-mini judge calls (one response each, in parallel)

PERSONALIZATION_METRIC = "Personalization & Contextual Adaptation"
COUNSELING_METRICS = [m for m in COMPOSITE_METRICS if m != PERSONALIZATION_METRIC]  # the 7

RAW_DIR = RESULTS_DIR / "raw_scores"
SUMMARY_DIR = RESULTS_DIR / "summary"


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def judge_responses_concurrent(responses, eval_config, max_workers=JUDGE_WORKERS):
    """Judge each (question, response) with its own GPT call, run in parallel.

    One API call per response (unchanged); a thread pool issues up to
    `max_workers` simultaneously. Results are returned in input order.
    """
    out: List[Optional[Dict]] = [None] * len(responses)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {
            ex.submit(gpt_evaluate_single, r["question"], r["response"], eval_config): i
            for i, r in enumerate(responses)
        }
        for fut in futures:
            out[futures[fut]] = fut.result()
    return out


def _mean(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def client_scores_from_samples(samples: List[Dict]) -> Dict[str, Optional[float]]:
    """From per-sample judge dicts, compute this client's counseling + personalization."""
    metric_means = {
        m: _mean([s.get(m) for s in samples])
        for m in COUNSELING_METRICS + [PERSONALIZATION_METRIC]
    }
    counseling_components = [metric_means[m] for m in COUNSELING_METRICS if metric_means[m] is not None]
    return {
        "counseling": _mean(counseling_components),
        "personalization": metric_means[PERSONALIZATION_METRIC],
        "n_samples": len(samples),
    }


def group_means(
    per_client: Dict[int, Dict[str, Optional[float]]],
    client_types: Dict[int, str],
) -> Dict[str, Dict[str, Optional[float]]]:
    groups = {
        "all": list(per_client.keys()),
        "iid": [c for c, t in client_types.items() if t == "iid"],
        "noniid": [c for c, t in client_types.items() if t == "noniid"],
    }
    out: Dict[str, Dict[str, Optional[float]]] = {}
    for gname, cids in groups.items():
        cids = [c for c in cids if c in per_client]
        if not cids:
            continue
        out[gname] = {
            "counseling": _mean([per_client[c]["counseling"] for c in cids]),
            "personalization": _mean([per_client[c]["personalization"] for c in cids]),
        }
    return out


# ---------------------------------------------------------------------------
# Batched generation (left-padded) — much faster than one sample at a time
# ---------------------------------------------------------------------------
def generate_responses_batched(model, tokenizer, dataset, device, max_new_tokens,
                               max_samples=None, batch_size=EVAL_BATCH_SIZE, source_max_len=1024):
    """Greedy-generate responses in left-padded batches. Returns [{question,response}]."""
    n = min(len(dataset), max_samples) if max_samples else len(dataset)
    questions = [str(dataset[i].get("input", "")) for i in range(n)]
    results = []
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "left"  # decoder generation needs left padding
    use_autocast = device == "cuda" and torch.cuda.is_available()
    try:
        for s in tqdm(range(0, n, batch_size), desc="Generating (batched)", leave=False):
            batch_q = questions[s:s + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                for q in batch_q
            ]
            enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                            max_length=source_max_len).to(device)
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
                    out = model.generate(
                        **enc, max_new_tokens=max_new_tokens, do_sample=False,
                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                    )
            gen = out[:, enc["input_ids"].shape[1]:]
            for q, row in zip(batch_q, gen):
                results.append({"question": q, "response": tokenizer.decode(row, skip_special_tokens=True)})
    finally:
        tokenizer.padding_side = old_side
    return results


# ---------------------------------------------------------------------------
# Model reconstruction from a run's manifest
# ---------------------------------------------------------------------------
def load_client_model(config: ExperimentConfig, run_dir: Path, entry: Dict, gpu_id: int = 0):
    if entry.get("inference") == "fused":
        # FDLoRA: cross-blended fusion of global + personalized via forward hooks.
        from experiments.v2.fusion import build_fused_eval_model
        return build_fused_eval_model(config, run_dir, entry, gpu_id)
    model = get_model(config, gpu_id=gpu_id)
    primary = trainer.load_params(run_dir / entry["primary_file"])
    set_adapter_params(model, primary, adapter_name="default")
    if entry.get("local_file"):
        trainer.add_local_adapter_v2(model, config)
        local = trainer.load_params(run_dir / entry["local_file"])
        set_adapter_params(model, local, adapter_name="local")
        activate_adapters(model, ["default", "local"])
    else:
        activate_adapters(model, "default")
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Efficiency from metrics.jsonl
# ---------------------------------------------------------------------------
def efficiency_from_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return {"upload_mb": None, "compute_time_s": None, "peak_gpu_memory_mb": None}
    uploads, times, mems = [], [], []
    with open(path) as f:
        for line in f:
            entry = json.loads(line)
            for m in entry.get("client_metrics", {}).values():
                if m.get("upload_bytes") is not None:
                    uploads.append(m["upload_bytes"] / 1e6)
                if m.get("compute_time_s") is not None:
                    times.append(m["compute_time_s"])
                if m.get("peak_gpu_memory_mb") is not None:
                    mems.append(m["peak_gpu_memory_mb"])
    return {
        "upload_mb": _mean(uploads),
        "compute_time_s": _mean(times),
        "peak_gpu_memory_mb": _mean(mems),
    }


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------
def evaluate_run(
    method: str,
    seed: int,
    gpu_id: int = 0,
    force: bool = False,
    config: Optional[ExperimentConfig] = None,
    raw_dir: Optional[Path] = None,
    exp_label: Optional[str] = None,
) -> Optional[Dict]:
    # Defaults reproduce the canonical main-v2 behaviour exactly:
    #   config = build_config(method, seed)  -> save_dir == RESULTS_DIR
    #   raw_dir = RAW_DIR (results/raw_scores), exp_label = experiment_name
    # Ablations pass a pre-built (overridden) config plus a centralized raw_dir
    # and a setting-prefixed label so per-setting raw files never collide.
    config = config or build_config(method, seed)
    run_dir = Path(config.save_dir) / config.experiment_name
    raw_dir = raw_dir or RAW_DIR
    label = exp_label or config.experiment_name
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[eval] skip {label}: no manifest (not trained yet).")
        return None

    manifest = json.loads(manifest_path.read_text())
    client_data = prepare_all_clients(config)
    client_types = {cid: cd["type"] for cid, cd in client_data.items()}
    tokenizer, _ = get_tokenizer_and_data_collator(config)
    device = "cuda"

    raw_dir.mkdir(parents=True, exist_ok=True)
    per_client: Dict[int, Dict[str, Optional[float]]] = {}

    for cid_str, entry in manifest["clients"].items():
        cid = int(cid_str)
        raw_path = raw_dir / f"{label}_client{cid}.jsonl"
        if raw_path.exists() and not force:
            samples = [json.loads(l) for l in open(raw_path)]
        else:
            test_ds = client_data[cid]["test"]
            model = load_client_model(config, run_dir, entry, gpu_id)
            responses = generate_responses_batched(
                model, tokenizer, test_ds, device,
                max_new_tokens=config.eval.max_new_tokens,
                max_samples=config.eval.max_eval_samples,
                source_max_len=config.train.source_max_len,
            )
            del model
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            scores_list = judge_responses_concurrent(responses, config.eval)
            samples = [
                {"client_id": cid, "question": item["question"],
                 "response": item["response"], **scores}
                for item, scores in zip(responses, scores_list)
            ]
            with open(raw_path, "w") as f:
                for s in samples:
                    f.write(json.dumps(s) + "\n")

        per_client[cid] = client_scores_from_samples(samples)
        c = per_client[cid]
        print(f"[eval] {method} seed{seed} client {cid} ({client_types[cid]}): "
              f"counseling={c['counseling']}, personalization={c['personalization']}")

    groups = group_means(per_client, client_types)
    efficiency = efficiency_from_metrics(run_dir)

    # Per-client score log
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOGS_DIR / f"{label}_client_scores.jsonl", "w") as f:
        for cid in sorted(per_client):
            f.write(json.dumps({"client_id": cid, "type": client_types[cid],
                                **per_client[cid]}) + "\n")

    return {"method": method, "seed": seed, "groups": groups, "efficiency": efficiency,
            "per_client": {str(c): per_client[c] for c in per_client}}


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
def write_summary(results: List[Dict]) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary = {f"{r['method']}_seed{r['seed']}": r for r in results}
    with open(SUMMARY_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    csv_path = SUMMARY_DIR / "summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "seed", "group", "counseling", "personalization",
                    "upload_mb", "compute_time_s", "peak_gpu_memory_mb"])
        for r in results:
            eff = r["efficiency"]
            for group, vals in r["groups"].items():
                w.writerow([
                    r["method"], r["seed"], group,
                    vals.get("counseling"), vals.get("personalization"),
                    eff["upload_mb"], eff["compute_time_s"], eff["peak_gpu_memory_mb"],
                ])
    print(f"\n[eval] summary -> {SUMMARY_DIR / 'summary.json'} and {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate v2 runs (full test set, GPT-4o-mini)")
    parser.add_argument("--methods", nargs="+", default=list(V2_METHODS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(V2_SEEDS))
    parser.add_argument("--force", action="store_true", help="re-judge even if raw scores exist")
    parser.add_argument("--no-summary", action="store_true",
                        help="generate raw scores only; skip writing summary (avoids races "
                             "when running parallel split-eval processes)")
    args = parser.parse_args()

    results = []
    for method in args.methods:
        for seed in args.seeds:
            r = evaluate_run(method, seed, force=args.force)
            if r is not None:
                results.append(r)
    if results and not args.no_summary:
        write_summary(results)


if __name__ == "__main__":
    main()
