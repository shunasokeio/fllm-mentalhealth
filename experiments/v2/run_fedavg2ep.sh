#!/usr/bin/env bash
# Compute-matched control: train fedavg_2ep (2 epochs/round) for both seeds in
# parallel, evaluate, then regenerate the full summary including the new arm.
cd /home/mdx-user01/projects/fllm || exit 1
LOG=experiments/v2/logs
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[fedavg2ep $(ts)] === TRAIN (both seeds, GPUs 0+1) ==="
python -m experiments.v2.orchestrate --methods fedavg_2ep --seeds 42 99 --gpus 0 1

echo "[fedavg2ep $(ts)] === EVAL (batched gen + concurrent judge) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods fedavg_2ep --seeds 42 --no-summary > "$LOG/eval_f2ep_42.out" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.evaluate_v2 \
    --methods fedavg_2ep --seeds 99 --no-summary > "$LOG/eval_f2ep_99.out" 2>&1 &
P1=$!
wait $P0 || echo "[fedavg2ep] eval seed42 failed"
wait $P1 || echo "[fedavg2ep] eval seed99 failed"

echo "[fedavg2ep $(ts)] === REGEN FULL SUMMARY (all arms, cached) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods fedavg fedavg_2ep local_only selective dual_lora ha_duallora --seeds 42 99 \
    || echo "[fedavg2ep] summary failed"
echo "[fedavg2ep $(ts)] === DONE ==="
