"""v2 experiment configuration.

Builds a legacy ``ExperimentConfig`` (so the reused primitives in
``experiments.heterogeneity_aware_pfl`` accept it) with the v2 study's *fixed*
hyperparameters:

  * constant LR 2e-5 (never adapted, never decayed)
  * 1 epoch of local training per round (per-phase epochs can be overridden by a
    method, e.g. Method 4)
  * LoRA rank 16, all-linear, for both the global and local adapters
  * 10 rounds; 3 IID clients (alpha=100) + 7 non-IID clients (alpha=0.01)
  * no validation set (train 0.90 / test 0.10)
  * GPT-4o-mini judge on the *full* test set (max_eval_samples=None)

Also provides YAML dump/load so every launched run records the exact config it
ran with under ``experiments/v2/configs/``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict

import yaml

from experiments.heterogeneity_aware_pfl.config import (
    EvalConfig,
    ExperimentConfig,
    FLConfig,
    LocalLoRAConfig,
    LoRAConfig,
    ModelConfig,
    TrainConfig,
)

# Gradient checkpointing is disabled for v2: the 0.5B 4-bit model uses a small
# fraction of the 40GB GPUs, so we trade memory for ~2x throughput. Results are
# numerically identical (checkpointing only recomputes activations).
V2_GRADIENT_CHECKPOINTING = False

# ---------------------------------------------------------------------------
# Canonical v2 constants (fixed across ALL methods unless a method overrides)
# ---------------------------------------------------------------------------
V2_LR = 2e-5
V2_NUM_ROUNDS = 10
V2_LOCAL_EPOCHS = 1
V2_LORA_RANK = 16
V2_LORA_ALPHA = 8  # scaling = alpha / rank = 0.5 (matches the legacy fair setup)
V2_SEEDS = [42, 99]
V2_METHODS = ["fedavg", "local_only", "dual_lora", "ha_duallora", "selective"]

V2_DIR = Path(__file__).resolve().parent
RESULTS_DIR = V2_DIR / "results"
CONFIGS_DIR = V2_DIR / "configs"
LOGS_DIR = V2_DIR / "logs"


def build_config(
    method: str,
    seed: int,
    *,
    num_iid_clients: int = 3,
    num_noniid_clients: int = 7,
    save_dir: str | None = None,
    experiment_name: str | None = None,
) -> ExperimentConfig:
    """Construct the canonical v2 ExperimentConfig for a (method, seed) run.

    The keyword-only arguments are *overrides* for ablation studies; their
    defaults reproduce the canonical main-v2 setting byte-for-byte, so calling
    ``build_config(method, seed)`` is unchanged. Overrides let an ablation vary
    the IID/Non-IID client split and redirect outputs (``save_dir`` /
    ``experiment_name``) without touching any main result or config.
    """
    lora = LoRAConfig(rank=V2_LORA_RANK, alpha=V2_LORA_ALPHA, target_modules="all-linear")
    lora_local = LocalLoRAConfig(rank=V2_LORA_RANK, alpha=V2_LORA_ALPHA, target_modules="all-linear")

    train = TrainConfig(
        batch_size=1,
        gradient_accumulation_steps=4,
        optimizer="paged_adamw_8bit",
        # Constant LR: lr_max == lr_min == 2e-5. The v2 trainer uses a HF
        # "constant" scheduler with 0 warmup, so the effective LR is exactly
        # V2_LR every step of every round, for every adapter and client.
        lr_max=V2_LR,
        lr_min=V2_LR,
        local_epochs=V2_LOCAL_EPOCHS,
        max_grad_norm=0.3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        bf16=True,
    )

    fl = FLConfig(
        num_clients=num_iid_clients + num_noniid_clients,
        num_iid_clients=num_iid_clients,
        num_noniid_clients=num_noniid_clients,
        alpha_iid=100.0,
        alpha_noniid=0.01,
        num_rounds=V2_NUM_ROUNDS,
        num_gpus=1,  # process-level parallelism is handled by orchestrate.py
        train_fraction=0.90,
        val_fraction=0.0,  # no validation set; test indices are unchanged (see plan)
        test_fraction=0.10,
        seed=seed,
    )

    # Cap evaluation at 50 test samples per client (deterministic first-50).
    eval_cfg = EvalConfig(model="gpt-4o-mini", max_eval_samples=50, max_new_tokens=512)

    return ExperimentConfig(
        model=ModelConfig(gradient_checkpointing=V2_GRADIENT_CHECKPOINTING),
        lora=lora,
        lora_local=lora_local,
        train=train,
        fl=fl,
        eval=eval_cfg,
        experiment_name=experiment_name or f"{method}_seed{seed}",
        save_dir=save_dir or str(RESULTS_DIR),
    )


# ---------------------------------------------------------------------------
# YAML serialization
# ---------------------------------------------------------------------------
def dump_config_yaml(config: ExperimentConfig, path: Path) -> None:
    """Serialize the full ExperimentConfig to a YAML file for provenance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(dataclasses.asdict(config), f, sort_keys=False)


_SECTION_TYPES = {
    "model": ModelConfig,
    "lora": LoRAConfig,
    "lora_local": LocalLoRAConfig,
    "train": TrainConfig,
    "fl": FLConfig,
    "eval": EvalConfig,
}


def load_config_yaml(path: Path) -> ExperimentConfig:
    """Reconstruct an ExperimentConfig from a YAML file written by dump_config_yaml."""
    with open(path) as f:
        data: Dict[str, Any] = yaml.safe_load(f)

    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key in _SECTION_TYPES and isinstance(value, dict):
            kwargs[key] = _SECTION_TYPES[key](**value)
        else:
            kwargs[key] = value
    # lora_fedalt is unused by v2 but present in ExperimentConfig; let it default.
    kwargs.pop("lora_fedalt", None)
    return ExperimentConfig(**kwargs)
