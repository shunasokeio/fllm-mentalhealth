#!/usr/bin/env bash
# v2 eval-only pipeline (training already complete): base refs -> method eval
# (2-GPU split, batched generation, 50 samples/client) -> final summary.
cd /home/mdx-user01/projects/fllm || exit 1
LOG=experiments/v2/logs
mkdir -p "$LOG"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[run_eval $(ts)] === BASE references (50 samples, batched, both GPUs) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.run_base --seed 42 > "$LOG/base42.out" 2>&1 &
B0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.run_base --seed 99 > "$LOG/base99.out" 2>&1 &
B1=$!
wait $B0 || echo "[run_eval] base42 failed"
wait $B1 || echo "[run_eval] base99 failed"

echo "[run_eval $(ts)] === METHOD EVAL (full judge, 2-GPU split) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods fedavg dual_lora selective --seeds 42 99 --no-summary > "$LOG/eval_gpu0.out" 2>&1 &
P0=$!
CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.evaluate_v2 \
    --methods local_only ha_duallora --seeds 42 99 --no-summary > "$LOG/eval_gpu1.out" 2>&1 &
P1=$!
wait $P0 || echo "[run_eval] eval gpu0 group failed"
wait $P1 || echo "[run_eval] eval gpu1 group failed"

echo "[run_eval $(ts)] === FINAL SUMMARY (cached raw scores) ==="
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 || echo "[run_eval] final summary failed"

echo "[run_eval $(ts)] === ALL DONE ==="
