#!/usr/bin/env bash
# Compute-matched (1-epoch budget) controls: dual_lora_half + ha_duallora_half,
# both seeds. Train (2-at-a-time on GPUs 0/1) -> eval -> regenerate full summary
# so every method is compared at ~1 epoch/client/round.
cd /home/mdx-user01/projects/fllm || exit 1
LOG=experiments/v2/logs
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[halfbudget $(ts)] === TRAIN dual_lora_half + ha_duallora_half (GPUs 0+1) ==="
python -m experiments.v2.orchestrate --methods dual_lora_half ha_duallora_half --seeds 42 99 --gpus 0 1

echo "[halfbudget $(ts)] === EVAL (batched gen + concurrent judge) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods dual_lora_half --seeds 42 99 --no-summary > "$LOG/eval_dualhalf.out" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.evaluate_v2 \
    --methods ha_duallora_half --seeds 42 99 --no-summary > "$LOG/eval_hahalf.out" 2>&1 &
P1=$!
wait $P0 || echo "[halfbudget] eval dual_lora_half failed"
wait $P1 || echo "[halfbudget] eval ha_duallora_half failed"

echo "[halfbudget $(ts)] === REGEN FULL SUMMARY (all arms, cached) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods fedavg local_only selective dual_lora ha_duallora dual_lora_half ha_duallora_half \
    --seeds 42 99 || echo "[halfbudget] summary failed"
echo "[halfbudget $(ts)] === DONE ==="
