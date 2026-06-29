#!/usr/bin/env bash
# v2 full pipeline: train (2 GPU) -> base refs -> eval (2 GPU split) -> final summary.
# Crash-resilient: each stage tolerates failures and continues; orchestrate logs
# per-run errors to logs/<run>.err and keeps going.
cd /home/mdx-user01/projects/fllm || exit 1
LOG=experiments/v2/logs
mkdir -p "$LOG"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[run_all $(ts)] === STAGE 1: TRAINING (orchestrate, GPUs 0+1) ==="
python -m experiments.v2.orchestrate --gpus 0 1
echo "[run_all $(ts)] training stage finished"

echo "[run_all $(ts)] === STAGE 2: BASE references ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.run_base --seed 42 --reuse || echo "[run_all] base seed42 failed"
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.run_base --seed 99            || echo "[run_all] base seed99 failed"

echo "[run_all $(ts)] === STAGE 3: EVAL (full test set, GPT-4o-mini, 2-GPU split) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods fedavg dual_lora selective --seeds 42 99 --no-summary > "$LOG/eval_gpu0.out" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.evaluate_v2 \
    --methods local_only ha_duallora --seeds 42 99 --no-summary > "$LOG/eval_gpu1.out" 2>&1 &
P1=$!
wait $P0 || echo "[run_all] eval gpu0 group failed"
wait $P1 || echo "[run_all] eval gpu1 group failed"
echo "[run_all $(ts)] eval (gpu split) finished"

echo "[run_all $(ts)] === STAGE 4: FINAL SUMMARY (cached raw scores) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 || echo "[run_all] final summary failed"

echo "[run_all $(ts)] === ALL DONE ==="
