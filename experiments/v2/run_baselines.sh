#!/usr/bin/env bash
# Train + evaluate the FFA-LoRA baseline for seeds 42 99, using BOTH GPUs
# (seed 42 -> GPU 0, seed 99 -> GPU 1) for both training and eval.
#
# FFA-LoRA = FedAvg with lora_A frozen (B-only communication); it runs through the
# canonical v2 server. The final summary is rebuilt over ALL v2 methods from cached
# raw scores so FFA-LoRA is added without dropping any existing rows.
#
# NOTE: FDLoRA is implemented (experiments/v2/fusion.py) but intentionally NOT run
# as a baseline for this study — it is cited in related work instead.
cd /home/mdx-user01/projects/fllm || exit 1
LOG=experiments/v2/logs
mkdir -p "$LOG"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[run_baselines $(ts)] === FFA-LoRA TRAIN (2-GPU: seed42->GPU0, seed99->GPU1) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.server_v2 --method ffa_lora --seed 42 \
    > "$LOG/ffa_lora_seed42.out" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.server_v2 --method ffa_lora --seed 99 \
    > "$LOG/ffa_lora_seed99.out" 2>&1 &
P1=$!
wait $P0 || echo "[run_baselines] ffa_lora seed42 failed"
wait $P1 || echo "[run_baselines] ffa_lora seed99 failed"

echo "[run_baselines $(ts)] === FFA-LoRA EVAL (2-GPU: seed42->GPU0, seed99->GPU1) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods ffa_lora --seeds 42 --no-summary > "$LOG/eval_ffa_42.out" 2>&1 &
E0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.evaluate_v2 \
    --methods ffa_lora --seeds 99 --no-summary > "$LOG/eval_ffa_99.out" 2>&1 &
E1=$!
wait $E0 || echo "[run_baselines] eval ffa seed42 failed"
wait $E1 || echo "[run_baselines] eval ffa seed99 failed"

echo "[run_baselines $(ts)] === SUMMARY (all methods, cached raw scores) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods fedavg local_only dual_lora dual_lora_half ha_duallora ha_duallora_half \
              selective ffa_lora \
    --seeds 42 99 || echo "[run_baselines] summary failed"

echo "[run_baselines $(ts)] === ALL DONE ==="
