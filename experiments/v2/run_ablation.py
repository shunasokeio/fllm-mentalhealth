"""CLI driver for a single ablation setting (train and/or eval, then summarize).

    # whole setting on one GPU (train -> eval each method, then compile summary)
    CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.run_ablation --setting ratio_7iid_3noniid

    # one method / one phase (used for finer scheduling)
    CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.run_ablation \
        --setting ratio_1iid_9noniid --method fedavg --phase train

    # cross-setting ratio comparison table (reads main summary as the Mixed midpoint)
    python -m experiments.v2.run_ablation --ratio-comparison

Pass --gpu 0 (the index *within* the process; CUDA_VISIBLE_DEVICES pins the real GPU).
"""

from __future__ import annotations

import argparse

from experiments.v2 import ablations


def main() -> None:
    p = argparse.ArgumentParser(description="Run a v2 ablation setting")
    p.add_argument("--setting", choices=sorted(ablations.SETTINGS))
    p.add_argument("--method", default=None,
                   help="single method; default = all methods in the setting")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--phase", choices=["train", "eval", "both"], default="both")
    p.add_argument("--force", action="store_true", help="re-judge even if raw scores exist")
    p.add_argument("--ratio-comparison", action="store_true",
                   help="emit summary/ratio_comparison.json and exit")
    args = p.parse_args()

    if args.ratio_comparison:
        ablations.ratio_comparison(args.seed)
        return

    if not args.setting:
        p.error("--setting is required unless --ratio-comparison is given")

    methods = [args.method] if args.method else ablations.SETTINGS[args.setting]["methods"]
    for method in methods:
        if args.phase in ("train", "both"):
            ablations.train_one(args.setting, method, args.seed, args.gpu)
        if args.phase in ("eval", "both"):
            ablations.eval_one(args.setting, method, args.seed, args.gpu, force=args.force)

    # Compile the per-setting summary when the whole setting was run here.
    if not args.method:
        ablations.compile_summary(args.setting, args.seed)


if __name__ == "__main__":
    main()
