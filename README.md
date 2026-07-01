# Mental Health Counseling LLMs in Mixed IID/Non-IID Federations

**Shun Aso** — Faculty of Environment and Information Studies, Keio University, Fujisawa, Kanagawa, Japan
(`shun_aso@keio.jp`)



We study federated fine-tuning of a small LLM for mental health counseling when clients hold data of varying heterogeneity — a realistic scenario where some services draw from general populations (IID) while others serve narrow subpopulations (non-IID). Three research questions drive the study:

- **RQ1** Which federated method produces the best counseling quality overall?
- **RQ2** Can we reduce compute and communication by adapting training intensity to each client's data heterogeneity?
- **RQ3** For which clients does personalization help, and when does it hurt?

---

## Methods

All methods fine-tune **Qwen2.5-0.5B** (4-bit, bfloat16) with LoRA rank 16 across 10 rounds, 10 clients (3 IID + 7 non-IID):

| ID | Name | Description |
|---|---|---|
| M1 | **FedAvg** | Single global LoRA, standard FedAvg aggregation |
| M2 | **Local-Only** | Independent per-client LoRA, no aggregation |
| M3 | **DualLoRA** | Fixed dual-adapter: 1 epoch local then 1 epoch global per round |
| M4 | **HA-DualLoRA** *(proposed)* | Dual-adapter with per-phase epochs scaled by each client's entropy het-score |
| M5 | **Selective** | Otsu-binarized het-score: non-IID-like clients skip global aggregation |
| — | **FFA-LoRA** | Freeze-A LoRA communication-efficiency baseline |

### Heterogeneity score

Each client's static entropy het-score is computed once before training from its training data's cluster-label distribution:

```
p_k         = proportion of cluster k in client i's training data
H_i         = −∑_k p_k ln p_k
H_max       = ln(num_clusters)
het_score_i = 1 − H_i / H_max    ∈ [0, 1]   (→1 concentrated, →0 uniform)
```

M4 (HA-DualLoRA) scales each adapter's training budget continuously by this score. M5 (Selective) binarizes it via Otsu thresholding.

---

## Setup

Requires Python ≥ 3.11 and CUDA. We use [uv](https://github.com/astral-sh/uv) for dependency management.

```bash
git clone <repo-url>
cd <repo-name>
uv sync --frozen
```

Set your OpenAI key (needed for evaluation only):

```bash
export OPENAI_API_KEY=sk-...
```

---

## Data

We use [MentalChat16K](https://huggingface.co/datasets/ShenLab/MentalChat16K) (Apache 2.0). The dataset is downloaded automatically from HuggingFace when running the pipeline.

### Clustering pipeline

Topic clusters are used to simulate realistic non-IID data distributions across clients.

**Step 1 — Embed** each conversation `input` field with `meta-llama/Llama-3.1-8B-Instruct` (mean-pooled last hidden state, `max_length=512`):

```bash
python -m src.fllm.embedding
# → embedded_data/embeddings.npy  (shape: 16068 × 4096)
```

**Step 2 — Cluster** with HDBSCAN: L2-normalize embeddings → UMAP dimensionality reduction → HDBSCAN:

```bash
python -m src.fllm.clustering --reduce-dim umap
# → embedded_data/cluster_labels.npy   (shape: 16068,  values: 0–18)
# → embedded_data/clustered_dataset.csv  (cluster_id, input, output)
```

HDBSCAN produces 19 topic clusters (IDs 0–18) with no noise points. The resulting `cluster_labels.npy` is a numpy integer array of length 16,068 (one entry per MentalChat16K example).

### FL client splits

10 simulated clients: **3 IID** (Dirichlet α=100) + **7 non-IID** (Dirichlet α=0.01). Splits are reproducibly seeded (seeds 42 and 99 for the two paper runs).

---

## Reproducing Results

### 1. Train all methods (2 GPUs, 10 rounds each)

```bash
python -m experiments.v2.orchestrate --gpus 0 1
```

Single method/seed:

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.server_v2 --method fedavg --seed 42
```

Available methods: `fedavg`, `local_only`, `dual_lora`, `ha_duallora`, `selective`, `ffa_lora`

### 2. Compute base-model reference scores

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.run_base --seed 42 --reuse
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.run_base --seed 99
```

### 3. Evaluate with GPT-4o-mini judge (50 test samples per client)

```bash
# 2-GPU parallel eval
CUDA_VISIBLE_DEVICES=0 python -m experiments.v2.evaluate_v2 \
    --methods fedavg dual_lora selective --seeds 42 99 --no-summary &
CUDA_VISIBLE_DEVICES=1 python -m experiments.v2.evaluate_v2 \
    --methods local_only ha_duallora --seeds 42 99 --no-summary
# Aggregate summary table + figures
python -m experiments.v2.evaluate_v2
```

Or run the full pipeline end-to-end:

```bash
bash experiments/v2/run_all.sh
```

Outputs are written to `experiments/v2/results/summary/` (JSON, CSV, figures).

### 4. Ablations

```bash
python -m experiments.v2.orchestrate_ablations
python -m experiments.v2.run_ablation
```

### Evaluation rubric

Responses are scored by GPT-4o-mini (temperature 0) on 8 dimensions:

| Dimension | Reported as |
|---|---|
| Overall Rating | — |
| Active Listening | counseling quality |
| Empathy & Validation | counseling quality |
| Safety & Trustworthiness | counseling quality |
| Open-mindedness & Non-judgment | counseling quality |
| Clarity & Encouragement | counseling quality |
| Boundaries & Ethical | counseling quality |
| Holistic Approach | counseling quality |
| Personalization & Contextual Adaptation | standalone (not folded in) |

**Counseling quality** = mean of the 7 counseling dimensions. **Personalization** is reported separately.

---

## Results

Pre-computed evaluation outputs (per-sample judge scores and summary tables/figures) are in `experiments/v2/results/`. Trained adapter weights are large and available separately on HuggingFace Hub: [TODO].

---

## Repository Structure

```
src/fllm/
  embedding.py              Llama-3.1-8B embedding of MentalChat16K inputs
  clustering.py             UMAP + HDBSCAN topic clustering
  datasplitter.py           Dirichlet FL client splits
  fed.py                    FedAvg aggregation utilities
  eval.py                   GPT-4o-mini judge (8-metric rubric)
  eval_average.py           Aggregation helpers
  significance.py           Statistical significance tests
  cent.py                   Centralized (non-FL) fine-tuning
  generate.py               Response generation utilities
  qualitative_eval.py       Qualitative evaluation helpers
  heterogeneity_logging.py  Client heterogeneity logging

experiments/heterogeneity_aware_pfl/    Shared infrastructure (imported by v2)
  config.py                 Dataclass configs (ExperimentConfig, FLConfig, …)
  data_utils.py             Dataset loading and client data preparation
  evaluate.py               GPT-4o-mini judge prompt + per-sample scoring
  heterogeneity.py          Otsu client classification
  model_utils.py            LoRA adapter utilities (load, activate, aggregate)
  utils.py                  Logging, seeding helpers

experiments/v2/             Paper experiment suite
  server_v2.py              FL server: dispatches ClientPlans, aggregates adapters
  trainer.py                Local training (single and dual adapter)
  methods.py                Method registry (M1–M5, FFA-LoRA, ablations)
  het_score.py              Static entropy het-score + Otsu classification
  ffa_lora.py               FFA-LoRA freeze-A training
  fusion.py                 FDLoRA adapter fusion (related-work, not run)
  orchestrate.py            2-GPU crash-resilient job runner
  orchestrate_ablations.py  Ablation job runner
  evaluate_v2.py            GPT-4o-mini judge + summary reporting
  run_base.py               Base-model (no FL) reference scores
  ablations.py              Ablation study definitions
  run_ablation.py           Ablation evaluation
  v2_config.py              Canonical hyperparameters (LR, rounds, LoRA rank, seeds)
  plot_summary_v2.py        Summary figure generation
  plot_counseling_epoch_matched.py  Epoch-matched counseling plot
  configs/                  YAML configs for every (method, seed) run
  results/summary/          Paper tables and figures (JSON, CSV, PNG)
  results/raw_scores/       Per-sample judge scores (JSON)
  results/<method>_seed<N>/ Per-run training logs (metrics.jsonl, manifest.json)

pyproject.toml              Project dependencies
uv.lock                     Pinned dependency lockfile
```

---

## Citation 
[TODO]
```bibtex
```
