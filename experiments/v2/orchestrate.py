"""Crash-resilient 2-GPU orchestrator for the v2 training runs.

Builds a queue of (method, seed) jobs and runs them two-at-a-time, pinning each
to a GPU via CUDA_VISIBLE_DEVICES=0 / =1 (process-level parallelism — each run is
a separate `server_v2` subprocess that internally uses one GPU). If a run
crashes, its traceback is captured to ``logs/{run}.err`` and the queue
continues; the orchestrator never blocks on a single failure.

    python -m experiments.v2.orchestrate                      # all M1-M5 x seeds
    python -m experiments.v2.orchestrate --methods fedavg     # subset
    python -m experiments.v2.orchestrate --seeds 42 --gpus 0 1
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from experiments.v2.v2_config import LOGS_DIR, V2_METHODS, V2_SEEDS

Job = Tuple[str, int]  # (method, seed)


def build_queue(methods: List[str], seeds: List[int]) -> List[Job]:
    return [(m, s) for m in methods for s in seeds]


def launch(job: Job, gpu: int) -> subprocess.Popen:
    method, seed = job
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    err_path = LOGS_DIR / f"{method}_seed{seed}.err"
    err_file = open(err_path, "w")
    cmd = [sys.executable, "-m", "experiments.v2.server_v2",
           "--method", method, "--seed", str(seed), "--gpu", "0"]
    print(f"[orchestrate] launch {method} seed{seed} on GPU {gpu}: {' '.join(cmd)}", flush=True)
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=err_file)
    proc._v2_job = job        # type: ignore[attr-defined]
    proc._v2_gpu = gpu        # type: ignore[attr-defined]
    proc._v2_errfile = err_file  # type: ignore[attr-defined]
    return proc


def run(methods: List[str], seeds: List[int], gpus: List[int], poll_s: float = 5.0) -> None:
    queue = build_queue(methods, seeds)
    print(f"[orchestrate] {len(queue)} jobs across GPUs {gpus}: {queue}", flush=True)

    free_gpus = list(gpus)
    running: List[subprocess.Popen] = []
    failures: List[Job] = []
    completed: List[Job] = []

    while queue or running:
        # Fill free GPUs.
        while queue and free_gpus:
            gpu = free_gpus.pop(0)
            running.append(launch(queue.pop(0), gpu))

        time.sleep(poll_s)

        still: List[subprocess.Popen] = []
        for proc in running:
            ret = proc.poll()
            if ret is None:
                still.append(proc)
                continue
            job = proc._v2_job          # type: ignore[attr-defined]
            gpu = proc._v2_gpu          # type: ignore[attr-defined]
            proc._v2_errfile.close()    # type: ignore[attr-defined]
            free_gpus.append(gpu)
            if ret == 0:
                completed.append(job)
                print(f"[orchestrate] DONE {job} (GPU {gpu})", flush=True)
            else:
                failures.append(job)
                print(f"[orchestrate] FAILED {job} (exit {ret}) — see "
                      f"{LOGS_DIR / f'{job[0]}_seed{job[1]}.err'}; continuing.", flush=True)
        running = still

    print(f"\n[orchestrate] finished. {len(completed)} ok, {len(failures)} failed.", flush=True)
    if failures:
        print(f"[orchestrate] failures: {failures}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v2 training jobs across 2 GPUs")
    parser.add_argument("--methods", nargs="+", default=list(V2_METHODS),
                        help=f"default: {V2_METHODS}")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(V2_SEEDS))
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    args = parser.parse_args()
    run(args.methods, args.seeds, args.gpus)


if __name__ == "__main__":
    main()
