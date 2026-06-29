#!/usr/bin/env bash
# Method eval only (base refs already done): 2-GPU split, batched generation +
# concurrent judging, 50 samples/client. Cached client-evals are skipped/resumed.
cd /home/mdx-user01/projects/fllm || exit 1
LOG=experiments/v2/logs
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[run_eval_m $(ts)] === METHOD EVAL (batched gen + concurrent judge, 2-GPU split) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods fedavg dual_lora selective --seeds 42 99 --no-summary > "$LOG/eval_gpu0.out" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.evaluate_v2 \
    --methods local_only ha_duallora --seeds 42 99 --no-summary > "$LOG/eval_gpu1.out" 2>&1 &
P1=$!
wait $P0 || echo "[run_eval_m] eval gpu0 group failed"
wait $P1 || echo "[run_eval_m] eval gpu1 group failed"

echo "[run_eval_m $(ts)] === FINAL SUMMARY ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 || echo "[run_eval_m] summary failed"
echo "[run_eval_m $(ts)] === ALL DONE ==="
