# Mental Health Counseling LLMs in Mixed IID/Non-IID Federations

**Shun Aso** — Faculty of Environment and Information Studies, Keio University, Fujisawa, Kanagawa, Japan
(`shun_aso@keio.jp`)

[TODO]
> [[Paper PDF]]() · [[BibTeX]](#citation)

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
| M3 | **FFA-LoRA** | Freeze-A LoRA communication-efficiency baseline |
| M4 | **DualLoRA** | Fixed dual-adapter: 1 epoch local then 1 epoch global per round |
| M5 | **HA-DualLoRA** *(proposed)* | Dual-adapter with per-phase epochs scaled by each client's entropy het-score |
| M6 | **Selective** | Otsu-binarized het-score: non-IID-like clients skip global aggregation |

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
git clone https://github.com/shunasokeio/fllm-mentalhealth.git
cd fllm-mentalhealth
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

### LLM Judge

Responses are scored by **GPT-4o-mini** (temperature 0)

The judge scores 8 dimensions on a 1–10 scale:

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

**Counseling quality** = mean of the 7 counseling dimensions (metrics 1–7). **Personalization** is reported separately (metric 8 uses a distinct rubric).

<summary>Full judge system prompt</summary>

```
Please act as an impartial judge and evaluate the quality of the response provided by an AI mental health counseling assistant to the user question displayed below.
• Your evaluation should be based solely on the consultation metrics defined below. Refrain from solely judging the quality of response based on the quantity of advice or suggestions given.
• Avoid any potential bias and ensure that the order in which the responses were presented does not affect your judgment.
• Do not allow the length of the responses to influence your evaluation.
• Do not favor certain names of the assistants.
• Be as objective as possible.
• You must rate the response on a scale of 1 to 10 by strictly following the given format below.
• You must also rate the response in terms of EACH consultation metric defined below on a scale of 1 to 10 by strictly following the given format below.
• The ratings do not necessarily need to be the same.
• NOTE: Metrics 1–7 use the General Scoring Rubric below. Metric 8 (Personalization & Contextual Adaptation) uses its own distinct scoring rubric and must be scored accordingly.

IMPORTANT: Your output MUST strictly follow this format for every evaluation, so it can be parsed automatically:

Overall Rating: [number]
Active Listening: [number]
Empathy & Validation: [number]
Safety & Trustworthiness: [number]
Open-mindedness & Non-judgment: [number]
Clarity & Encouragement: [number]
Boundaries & Ethical: [number]
Holistic Approach: [number]
Personalization & Contextual Adaptation: [number]

Consultation Metrics:
Active Listening: Responses demonstrate careful consideration of user concerns, reflecting understanding and capturing the essence of the issue. Avoid assumptions or jumping to conclusions.
Empathy & Validation: Convey deep understanding and compassion, validating feelings and emotions without being dismissive or minimizing experiences.
Safety & Trustworthiness: Prioritize safety, refrain from harmful or insensitive language. Ensure the information provided is consistent and trustworthy.
Open-mindedness & Non-judgment: Approach without bias or judgment. Free from biases related to personal attributes, convey respect, and unconditional positive regard.
Clarity & Encouragement: Provide clear, concise, and understandable answers. Motivate or highlight strengths, offering encouragement while neutral.
Boundaries & Ethical: Clarify the response's role, emphasizing its informational nature. In complex scenarios, guide users to seek professional assistance.
Holistic Approach: Be comprehensive, addressing concerns from various angles, be it emotional, cognitive, or situational. Consider the broader context, even if not explicitly detailed in the query.
Personalization & Contextual Adaptation: The response is tailored to this specific user rather than a generic mental-health audience. It (a) explicitly acknowledges details of the user's stated situation, history, stressors, or relationship to the issue; (b) adapts tone, depth, vocabulary, and therapeutic framing to fit the user's apparent emotional state and prior context; (c) avoids generic, boilerplate, or one-size-fits-all advice (e.g., bare "try mindfulness," "talk to a professional") that could be sent verbatim to any other user; and (d) integrates information from the user's question rather than ignoring it. A response that is empathic but interchangeable across users should score in the middle of the scale; a response that is uniquely useful to this user should score high.

General Scoring Rubric (Metrics 1–7):
1: The response completely fails to address the metric, showing a total disregard for the user's needs or concerns in this area.
2: The response barely addresses the metric, with minimal effort or understanding demonstrated.
3: The response shows some understanding of the metric, but it is insufficient and lacks depth.
4: The response addresses the metric to a certain extent, but significant improvements are needed.
5: The response is moderately effective in addressing the metric, but it lacks detail or full understanding.
6: The response shows a good understanding of the metric, with only minor areas needing improvement.
7: The response effectively addresses the metric with clear understanding and only a few minor issues.
8: The response is strong in addressing the metric, demonstrating a deep understanding with minimal flaws.
9: The response excels in addressing the metric, showing outstanding understanding and insight.
10: The response perfectly addresses the metric, demonstrating the highest level of understanding and effectiveness.

Personalization & Contextual Adaptation Scoring Rubric (Metric 8 only):
1-2: Response is fully generic; no reference to the user's situation; could be copy-pasted to any user with the same topic.
3-4: Single shallow acknowledgment of topic (e.g., "I hear you're feeling anxious") but content is otherwise template-like.
5-6: Multiple references to user's stated context, but tone/depth/framing is still average; some boilerplate persists.
7-8: Response clearly draws on the user's specific stressors/wording, adapts tone and depth accordingly, and avoids boilerplate; a different user with a different situation would receive a meaningfully different response.
9-10: Response is densely contextualized — names specific user-stated facts, mirrors user's vocabulary/register, and selects a therapeutic framing that fits this user's apparent emotional state. Could not plausibly be sent to a different user.
```

User turn template:

```
[User Question]
{user_question}

[The Start of AI assistant's Answer]
{model_response}
[The End of AI assistant's Answer]
```

</details>

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

prompts/
  llm_judge_mentalchat.md   Full GPT-4o-mini judge system prompt (MentalChat16K)

pyproject.toml              Project dependencies
uv.lock                     Pinned dependency lockfile
```

---

## Citation

[TODO]
