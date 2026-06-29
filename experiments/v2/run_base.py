"""Base-model (no-LoRA) reference scores for v2.

Per the plan: the seed-42 base can be imported from the legacy
``results2/base_qwen_reeval`` (its per-client 8-metric means let us recompute
counseling vs personalization without re-judging), while the seed-99 base must be
re-evaluated because its client partition / test split differ.

  CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.run_base --seed 99
  python -m experiments.v2.run_base --seed 42 --reuse   # import legacy seed-42 base

⚠️ Caveat: the legacy base was judged on max_eval_samples=20, whereas v2 methods
use the full test set. Pass --eval (no --reuse) to re-judge seed-42 base on the
full set for strict comparability.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

from experiments.heterogeneity_aware_pfl.config import ExperimentConfig
from experiments.heterogeneity_aware_pfl.data_utils import prepare_all_clients
from experiments.heterogeneity_aware_pfl.evaluate import gpt_evaluate_single
from experiments.heterogeneity_aware_pfl.model_utils import (
    get_tokenizer_and_data_collator,
    load_model_for_inference,
)

from experiments.v2.evaluate_v2 import (
    COUNSELING_METRICS,
    PERSONALIZATION_METRIC,
    RAW_DIR,
    client_scores_from_samples,
    generate_responses_batched,
    group_means,
)
from experiments.v2.v2_config import RESULTS_DIR, build_config

LEGACY_BASE_42 = (
    Path(__file__).resolve().parent.parent
    / "heterogeneity_aware_pfl" / "results2" / "base_qwen_reeval" / "test_individual.jsonl"
)


def _import_legacy_seed42(client_types: Dict[int, str]) -> Dict:
    """Recompute counseling/personalization from legacy per-client 8-metric means."""
    per_client: Dict[int, Dict[str, Optional[float]]] = {}
    for line in open(LEGACY_BASE_42):
        row = json.loads(line)
        cid = int(row["client_id"])
        counseling_vals = [row[m] for m in COUNSELING_METRICS if m in row]
        per_client[cid] = {
            "counseling": sum(counseling_vals) / len(counseling_vals) if counseling_vals else None,
            "personalization": row.get(PERSONALIZATION_METRIC),
            "n_samples": None,
        }
    return {
        "seed": 42, "source": "imported_legacy_base_qwen_reeval",
        "note": "legacy judged on 20 samples/client, not full test set",
        "groups": group_means(per_client, client_types),
        "per_client": {str(c): per_client[c] for c in per_client},
    }


def _eval_base(config: ExperimentConfig, gpu_id: int = 0) -> Dict:
    client_data = prepare_all_clients(config)
    client_types = {cid: cd["type"] for cid, cd in client_data.items()}
    model, device = load_model_for_inference(config)
    tokenizer, _ = get_tokenizer_and_data_collator(config)
    model.eval()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    per_client: Dict[int, Dict[str, Optional[float]]] = {}
    for cid in sorted(client_data):
        test_ds = client_data[cid]["test"]
        responses = generate_responses_batched(
            model, tokenizer, test_ds, device,
            max_new_tokens=config.eval.max_new_tokens,
            max_samples=config.eval.max_eval_samples,
            source_max_len=config.train.source_max_len,
        )
        samples = []
        for item in responses:
            scores = gpt_evaluate_single(item["question"], item["response"], config.eval)
            samples.append({"client_id": cid, **scores})
        with open(RAW_DIR / f"base_seed{config.fl.seed}_client{cid}.jsonl", "w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        per_client[cid] = client_scores_from_samples(samples)
        print(f"[base seed{config.fl.seed}] client {cid} ({client_types[cid]}): "
              f"counseling={per_client[cid]['counseling']}, "
              f"personalization={per_client[cid]['personalization']}")

    return {
        "seed": config.fl.seed, "source": "evaluated_full_test_set",
        "groups": group_means(per_client, client_types),
        "per_client": {str(c): per_client[c] for c in per_client},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Base-model reference scores for v2")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--reuse", action="store_true",
                        help="seed 42 only: import legacy base instead of re-evaluating")
    args = parser.parse_args()

    config = build_config("base", args.seed)
    client_types = {cid: cd["type"] for cid, cd in prepare_all_clients(config).items()}

    if args.reuse and args.seed == 42 and LEGACY_BASE_42.exists():
        result = _import_legacy_seed42(client_types)
    else:
        result = _eval_base(config)

    out_dir = RESULTS_DIR / f"base_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "base_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[base] seed {args.seed} -> {out_dir / 'base_results.json'}  ({result['source']})")
    for group, vals in result["groups"].items():
        print(f"  {group}: counseling={vals.get('counseling')}, "
              f"personalization={vals.get('personalization')}")


if __name__ == "__main__":
    main()
