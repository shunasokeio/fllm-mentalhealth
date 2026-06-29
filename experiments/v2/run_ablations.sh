#!/usr/bin/env bash
# Run all v2 ablations via the balanced 2-GPU orchestrator.
#
# Prerequisites:
#   export OPENAI_API_KEY=...        # GPT-4o-mini judge
#   HF access for EmoCareAI/Psych8k and meta-llama/Llama-3.1-8B-Instruct
#
# Phase A (MentalChat16K): all (setting, method) jobs for the three MentalChat
#   settings (ratio_7iid_3noniid, ratio_1iid_9noniid, ha_sched) are dispatched
#   across both GPUs via a shared queue — both GPUs stay busy the whole time.
# Phase B (Psych8k): the global data files are swapped in once, dataset_counselchat
#   runs across both GPUs, then MentalChat is restored.
# Each setting's het-score cache is precomputed serially first (no parallel races),
# its summary is compiled when its methods finish, and the ratio comparison runs last.
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root

python -m experiments.v2.orchestrate_ablations --gpus 0 1

echo "=== All ablations done. Summaries in experiments/v2/results/ablations/summary/ ==="
