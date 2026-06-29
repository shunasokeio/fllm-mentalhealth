"""Balanced 2-GPU orchestrator for the v2 ablations.

Phase A (MentalChat16K, read-only data): every (setting, method) job for the
three MentalChat settings is pushed onto one queue and dispatched to whichever
GPU is free, so both GPUs stay busy regardless of per-run cost. A setting's
summary is compiled as soon as its methods all finish.

Phase B (Psych8k): the global data files are swapped in once, the dataset
setting's methods are run across both GPUs, then MentalChat is restored.

Race-safety: each setting's het-score cache is computed *once*, serially, before
its jobs are dispatched, so parallel methods of the same setting never write the
same het_scores_seed*.json concurrently.

    python -m experiments.v2.orchestrate_ablations            # both phases
    python -m experiments.v2.orchestrate_ablations --gpus 0 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from experiments.v2 import ablations
from experiments.v2.het_score import get_scores_and_classification
from experiments.v2.v2_config import LOGS_DIR

Job = Tuple[str, str]  # (setting, method)

MENTALCHAT_SETTINGS = ["ratio_7iid_3noniid", "ratio_1iid_9noniid", "ha_sched"]
PSYCH8K_SETTINGS = ["dataset_counselchat"]


def precompute_het(settings: List[str]) -> None:
    """Populate each setting's het-score cache once (avoids parallel write races)."""
    for s in settings:
        m0 = ablations.SETTINGS[s]["methods"][0]
        cfg = ablations.build_ablation_config(s, m0)
        get_scores_and_classification(cfg)
        print(f"[orch] het cache ready: {s}", flush=True)


def jobs_for(settings: List[str]) -> List[Job]:
    return [(s, m) for s in settings for m in ablations.SETTINGS[s]["methods"]]


def run_queue(jobs: List[Job], gpus: List[int], poll_s: float = 5.0) -> List[Job]:
    """Dispatch (setting, method) jobs across GPUs; compile a setting when complete."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    queue = list(jobs)
    free = list(gpus)
    running: List[tuple] = []  # (proc, job, gpu, errfile)
    total = defaultdict(int)
    done = defaultdict(int)
    for s, _ in jobs:
        total[s] += 1
    failures: List[Job] = []

    print(f"[orch] {len(queue)} jobs across GPUs {gpus}: {queue}", flush=True)
    while queue or running:
        while queue and free:
            gpu = free.pop(0)
            setting, method = queue.pop(0)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # GPU re-indexed to 0 in-process
            errf = open(LOGS_DIR / f"orch_{setting}_{method}.log", "w")
            cmd = [sys.executable, "-m", "experiments.v2.run_ablation",
                   "--setting", setting, "--method", method, "--phase", "both", "--gpu", "0"]
            print(f"[orch] launch {setting}/{method} on GPU {gpu}", flush=True)
            proc = subprocess.Popen(cmd, env=env, stdout=errf, stderr=subprocess.STDOUT)
            running.append((proc, (setting, method), gpu, errf))

        time.sleep(poll_s)

        still = []
        for proc, job, gpu, errf in running:
            ret = proc.poll()
            if ret is None:
                still.append((proc, job, gpu, errf))
                continue
            errf.close()
            free.append(gpu)
            setting, method = job
            if ret == 0:
                done[setting] += 1
                print(f"[orch] DONE {setting}/{method} (GPU {gpu}) "
                      f"[{done[setting]}/{total[setting]}]", flush=True)
                if done[setting] == total[setting]:
                    print(f"[orch] compiling summary: {setting}", flush=True)
                    ablations.compile_summary(setting)
            else:
                failures.append(job)
                print(f"[orch] FAILED {setting}/{method} (exit {ret}); see "
                      f"{LOGS_DIR / f'orch_{setting}_{method}.log'}; continuing.", flush=True)
        running = still
    return failures


def _psych8k(cmd_args: List[str]) -> None:
    subprocess.run([sys.executable, "-m", "experiments.v2.prepare_psych8k", *cmd_args], check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Balanced 2-GPU ablation orchestrator")
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    gpus = args.gpus

    # ----- Phase A: MentalChat settings, both GPUs -----
    print("=== Phase A: MentalChat ablations (balanced queue) ===", flush=True)
    precompute_het(MENTALCHAT_SETTINGS)
    fail_a = run_queue(jobs_for(MENTALCHAT_SETTINGS), gpus)

    # ----- Phase B: Psych8k (isolated global data swap) -----
    print("=== Phase B: Psych8k dataset ablation (isolated) ===", flush=True)
    _psych8k([])            # build (cached -> no-op)
    _psych8k(["--swap-in"])  # backup MentalChat, activate Psych8k
    fail_b: List[Job] = []
    try:
        precompute_het(PSYCH8K_SETTINGS)  # het over the now-active Psych8k data
        fail_b = run_queue(jobs_for(PSYCH8K_SETTINGS), gpus)
    finally:
        _psych8k(["--restore"])  # always restore MentalChat

    # ----- Ratio comparison + report -----
    print("=== Ratio comparison ===", flush=True)
    ablations.ratio_comparison()

    failures = fail_a + fail_b
    if failures:
        print(f"=== DONE with failures: {failures} ===", flush=True)
    else:
        print("=== DONE: all ablation runs succeeded. "
              "Summaries in experiments/v2/results/ablations/summary/ ===", flush=True)


if __name__ == "__main__":
    main()
