"""Constant-LR training core for the v2 suite.

Two entry points, both training at a flat LR of ``config.train.lr_max`` (= 2e-5)
with a HF "constant" scheduler and zero warm-up, so the learning rate is exactly
the same every step, every round, for every adapter and client. No cosine decay,
no per-client sigma scaling, no proximal-L2 anchor (all of which the legacy
client hard-wires and the v2 spec forbids):

  * ``train_single_adapter`` — one LoRA adapter (Methods 1, 2; reserved for 7).
  * ``train_dual_phase``     — global ("default") + local ("local") adapters,
    trained in alternating phases with the inactive one frozen (Methods 3, 4, 5).

Adapter parameters are passed/returned as ``List[np.ndarray]`` so the server can
persist them and run FedAvg without holding GPU models.
"""

from __future__ import annotations

import gc
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from peft import LoraConfig
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import ProgressCallback

from experiments.heterogeneity_aware_pfl.config import ExperimentConfig
from experiments.heterogeneity_aware_pfl.model_utils import (
    activate_adapters,
    get_adapter_param_keys,
    get_adapter_params,
    get_model,
    get_tokenizer_and_data_collator,
    set_adapter_params,
)
from experiments.heterogeneity_aware_pfl.utils import (
    RoundTimer,
    get_peak_gpu_memory_mb,
    measure_serialization_bytes,
    reset_gpu_memory_tracking,
)

Params = List[np.ndarray]


# ---------------------------------------------------------------------------
# Parameter IO (shared by server + evaluator)
# ---------------------------------------------------------------------------
def save_params(params: Params, path) -> None:
    """Persist a list of numpy adapter arrays to a .npz file (ordered)."""
    np.savez(path, **{f"p{i}": p for i, p in enumerate(params)})


def load_params(path) -> Params:
    """Load adapter arrays previously saved by ``save_params``, in order."""
    data = np.load(path)
    return [data[f"p{i}"] for i in range(len(data.files))]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_target_modules(model) -> List[str]:
    """Leaf module names the 'default' adapter targets (e.g. q_proj, ...).

    The legacy ``add_local_adapter`` re-resolves "all-linear" on the already
    PEFT-wrapped base model, which finds the wrong set; deriving the names from
    the default adapter's own parameters guarantees the local adapter targets the
    exact same modules.
    """
    targets = set()
    for name, _ in model.named_parameters():
        if ".lora_A.default." in name:
            targets.add(name.split(".lora_A.default.")[0].split(".")[-1])
    return sorted(targets)


def add_local_adapter_v2(model, config: ExperimentConfig) -> None:
    """Add the 'local' LoRA adapter targeting the same modules as 'default'."""
    lc = config.lora_local
    local_config = LoraConfig(
        r=lc.rank,
        lora_alpha=lc.alpha,
        lora_dropout=lc.dropout,
        task_type="CAUSAL_LM",
        target_modules=_resolve_target_modules(model),
        bias=lc.bias,
    )
    model.add_adapter("local", local_config)


def _one_epoch_steps(n_examples: int, config: ExperimentConfig) -> int:
    eff_batch = max(1, config.train.batch_size * config.train.gradient_accumulation_steps)
    return max(1, math.ceil(n_examples / eff_batch))


# The global adapter is the PEFT "default" adapter; the personalized one is "local".
_ADAPTER_BY_PHASE = {"global": "default", "local": "local", "default": "default"}


def _set_trainable(model, adapter_name: str) -> None:
    """Enable grads only for the named adapter's LoRA params; freeze everything else."""
    tag = f".{adapter_name}."
    n_trainable = 0
    for name, param in model.named_parameters():
        on = tag in name
        param.requires_grad_(on)
        n_trainable += int(on)
    if n_trainable == 0:
        raise RuntimeError(f"No trainable params matched adapter '{adapter_name}'")


def _set_trainable_b_only(model) -> None:
    """FFA-LoRA: train only the 'default' adapter's lora_B; freeze lora_A and all else.

    Asserts (the FFA-LoRA validation) that every lora_A param is frozen and every
    lora_B param is trainable after the split.
    """
    n_b = 0
    for name, param in model.named_parameters():
        on = ".lora_B.default." in name
        param.requires_grad_(on)
        n_b += int(on)
    if n_b == 0:
        raise RuntimeError("FFA-LoRA: no lora_B.default params found to train")
    for name, param in model.named_parameters():
        if ".lora_A." in name:
            assert not param.requires_grad, f"FFA-LoRA: lora_A must be frozen ({name})"
        if ".lora_B.default." in name:
            assert param.requires_grad, f"FFA-LoRA: lora_B must be trainable ({name})"


def _training_args(config: ExperimentConfig, output_dir: str, max_steps: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config.train.batch_size,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        max_steps=max_steps,
        learning_rate=config.train.lr_max,          # constant 2e-5
        lr_scheduler_type="constant",
        warmup_ratio=0.0,                            # no warm-up => flat from step 0
        weight_decay=config.train.weight_decay,
        max_grad_norm=config.train.max_grad_norm,
        optim=config.train.optimizer,
        bf16=config.train.bf16,
        fp16=False,
        logging_steps=5,
        save_strategy="no",
        remove_unused_columns=False,
        gradient_checkpointing=config.model.gradient_checkpointing,
        group_by_length=False,
        disable_tqdm=True,
        report_to="none",
        seed=config.fl.seed,
    )


def _run_phase(
    model, tokenizer, collator, dataset, config: ExperimentConfig,
    train_adapter: str, max_steps: int, client_id: int, b_only: bool = False,
) -> float:
    """Train one phase (a single adapter) and return the mean training loss.

    ``b_only`` (FFA-LoRA): train only lora_B of the 'default' adapter, freezing A.
    """
    adapter = _ADAPTER_BY_PHASE.get(train_adapter, train_adapter)
    if b_only:
        _set_trainable_b_only(model)
    else:
        _set_trainable(model, adapter)
    args = _training_args(config, f"/tmp/v2_client_{client_id}_{adapter}", max_steps)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
        processing_class=tokenizer,
    )
    trainer.remove_callback(ProgressCallback)
    result = trainer.train()
    return float(result.metrics.get("train_loss", 0.0))


# ---------------------------------------------------------------------------
# Entry point 1: single-adapter training (Methods 1, 2)
# ---------------------------------------------------------------------------
def train_single_adapter(
    config: ExperimentConfig,
    dataset,
    primary_params: Params,
    epochs: float,
    gpu_id: int,
    client_id: int,
    freeze_a: bool = False,
) -> Tuple[Params, Dict]:
    """Train a single 'default' adapter for `epochs` epochs at constant LR.

    ``freeze_a`` (FFA-LoRA, Method 7): freeze lora_A and train only lora_B; the
    reported ``upload_bytes`` then count lora_B alone (A is never communicated).
    """
    reset_gpu_memory_tracking()
    model = get_model(config, gpu_id=gpu_id)
    # Required for grads to flow when only a deep adapter trains and gradient
    # checkpointing is off (prepare_model_for_kbit_training enables this only
    # when checkpointing is on). Idempotent and harmless otherwise.
    model.enable_input_require_grads()
    tokenizer, collator = get_tokenizer_and_data_collator(config)
    set_adapter_params(model, primary_params, adapter_name="default")
    activate_adapters(model, "default")

    n_examples = len(dataset)
    steps = max(1, round(_one_epoch_steps(n_examples, config) * epochs))

    with RoundTimer("train") as timer:
        train_loss = _run_phase(model, tokenizer, collator, dataset, config,
                                "default", steps, client_id, b_only=freeze_a)

    out = get_adapter_params(model, adapter_name="default")
    upload_bytes = (
        _measure_b_only_bytes(model, out) if freeze_a else measure_serialization_bytes(out)
    )
    metrics = {
        "train_loss": train_loss,
        "peak_gpu_memory_mb": get_peak_gpu_memory_mb(),
        "compute_time_s": timer.elapsed_seconds,
        "num_examples": n_examples,
        "global_steps": steps,
        "global_epochs": epochs,
        "upload_bytes": upload_bytes,
    }
    _cleanup(model)
    return out, metrics


def _measure_b_only_bytes(model, params: Params) -> int:
    """Serialized size of the lora_B params only (FFA-LoRA's true communication)."""
    keys = get_adapter_param_keys(model, adapter_name="default")
    b_params = [p for k, p in zip(keys, params) if "lora_B" in k]
    return measure_serialization_bytes(b_params)


# ---------------------------------------------------------------------------
# Entry point 2: dual-adapter phase training (Methods 3, 4, 5)
# ---------------------------------------------------------------------------
def train_dual_phase(
    config: ExperimentConfig,
    dataset,
    global_params: Params,
    local_params: Optional[Params],
    phases: Sequence[Tuple[str, float]],
    gpu_id: int,
    client_id: int,
) -> Tuple[Params, Params, Dict]:
    """Train global+local adapters in the given phases (each: adapter, epochs).

    Both adapters are active throughout; the phase's non-target adapter is frozen
    so its contribution is present but not updated. An adapter absent from
    `phases` (e.g. 'default' for a non-IID Selective client) is never trained and
    its returned params equal the inputs.
    """
    reset_gpu_memory_tracking()
    model = get_model(config, gpu_id=gpu_id)
    # Required for grads to flow when only a deep adapter trains and gradient
    # checkpointing is off (prepare_model_for_kbit_training enables this only
    # when checkpointing is on). Idempotent and harmless otherwise.
    model.enable_input_require_grads()
    tokenizer, collator = get_tokenizer_and_data_collator(config)
    set_adapter_params(model, global_params, adapter_name="default")
    add_local_adapter_v2(model, config)
    if local_params is not None:
        set_adapter_params(model, local_params, adapter_name="local")
    activate_adapters(model, ["default", "local"])

    n_examples = len(dataset)
    one_epoch = _one_epoch_steps(n_examples, config)

    losses: Dict[str, float] = {}
    phase_steps: Dict[str, int] = {}
    phase_epochs: Dict[str, float] = {}
    with RoundTimer("train") as timer:
        for adapter, epochs in phases:
            steps = max(1, round(one_epoch * epochs))
            losses[adapter] = _run_phase(model, tokenizer, collator, dataset, config,
                                         adapter, steps, client_id)
            phase_steps[adapter] = steps
            phase_epochs[adapter] = epochs

    global_out = get_adapter_params(model, adapter_name="default")
    local_out = get_adapter_params(model, adapter_name="local")
    metrics = {
        "train_loss": losses.get("global", losses.get("local", 0.0)),
        "train_loss_global": losses.get("global"),
        "train_loss_local": losses.get("local"),
        "peak_gpu_memory_mb": get_peak_gpu_memory_mb(),
        "compute_time_s": timer.elapsed_seconds,
        "num_examples": n_examples,
        "global_steps": phase_steps.get("global", 0),
        "local_steps": phase_steps.get("local", 0),
        "global_epochs": phase_epochs.get("global", 0.0),
        "local_epochs": phase_epochs.get("local", 0.0),
        "upload_bytes": measure_serialization_bytes(global_out),
    }
    _cleanup(model)
    return global_out, local_out, metrics


def _cleanup(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def initial_global_params(config: ExperimentConfig, gpu_id: int = 0) -> Params:
    """Build a single fresh 'default' adapter and return its params.

    Called once by the server so round-1 global params are identical for every
    client (LoRA B is zero-initialized, so this adapter is an identity at start).
    """
    model = get_model(config, gpu_id=gpu_id)
    params = get_adapter_params(model, adapter_name="default")
    _cleanup(model)
    return params
