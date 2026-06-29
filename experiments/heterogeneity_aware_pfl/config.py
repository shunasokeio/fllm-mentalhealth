"""All hyperparameters and experiment configurations."""

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
EMBEDDED_DIR = PROJECT_ROOT / "embedded_data"
CLUSTERED_CSV_PATH = EMBEDDED_DIR / "clustered_dataset.csv"
CLUSTER_LABELS_PATH = EMBEDDED_DIR / "cluster_labels.npy"

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    quantization: int = 4
    gradient_checkpointing: bool = True
    use_fast_tokenizer: bool = False


@dataclass
class LoRAConfig:
    rank: int = 32
    alpha: int = 16
    dropout: float = 0.075
    target_modules: str = "all-linear"
    bias: str = "none"


@dataclass
class LocalLoRAConfig:
    rank: int = 16
    alpha: int = 8
    dropout: float = 0.075
    target_modules: str = "all-linear"
    bias: str = "none"


@dataclass
class TrainConfig:
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    optimizer: str = "paged_adamw_8bit"
    lr_max: float = 0.0002
    lr_min: float = 0.0
    local_steps: int = 100
    local_epochs: int = 0  # if > 0, overrides local_steps with per-client epoch-based steps
    max_grad_norm: float = 0.3
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    source_max_len: int = 1024
    target_max_len: int = 256
    train_on_source: bool = False
    bf16: bool = True
    local_lr_scale: float = 0.5
    personalization_threshold: float = 0.3
    local_l2_mu: float = 0.01
    local_epoch_fraction: float = 1.0
    t_global: int = 50
    t_local_base: int = 50
    t_local_min: int = 15
    t_local_max: int = 75
    beta1: float = 0.3
    beta2: float = 0.5
    beta3: float = 0.4
    # local LR decays from lr_max*lr_factor down to lr_max*lr_factor*local_lr_min_factor
    # (instead of all the way to lr_min≈0), keeping the local adapter active in late rounds
    local_lr_min_factor: float = 0.5
    # if True, train local first then global (local captures client-specific variance first,
    # global update submitted for FedAvg is cleaner/more generalizable)
    local_first: bool = False


@dataclass
class FLConfig:
    num_clients: int = 10
    num_iid_clients: int = 3
    num_noniid_clients: int = 7
    alpha_iid: float = 100.0
    alpha_noniid: float = 0.01
    num_rounds: int = 20
    warmup_rounds: int = 2
    val_every: int = 2
    num_gpus: int = 1  # >1 → parallelize clients across GPUs within a round (process-based)
    train_fraction: float = 0.80
    val_fraction: float = 0.10
    test_fraction: float = 0.10
    fraction_fit: float = 1.0
    seed: int = 42


@dataclass
class FedALTLoRAConfig:
    rank: int = 16                      # 2 experts × rank16 = rank32 total, matching baseline1
    alpha: int = 8                      # scaling = alpha / rank = 0.5, same as baseline1
    dropout: float = 0.075              # match baseline1
    target_modules: str = "all-linear"  # match baseline1
    bias: str = "none"
    # Scalar-gate knobs. gate_init sets the logit so α=sigmoid(gate_init): -0.847 → α≈0.3,
    # making clients lean on the global/RoW expert by default and earn local weight only
    # when local training justifies it. gate_lr_mult scales the gate's LR vs the matrix LR
    # (a single scalar needs a much larger step to move within the few rounds we train).
    gate_init: float = -0.847           # sigmoid(-0.847) ≈ 0.30
    gate_lr_mult: float = 30.0


@dataclass
class EvalConfig:
    model: str = "gpt-4o-mini"
    max_eval_samples: int = 20
    max_new_tokens: int = 512


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    lora_local: LocalLoRAConfig = field(default_factory=LocalLoRAConfig)
    lora_fedalt: FedALTLoRAConfig = field(default_factory=FedALTLoRAConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fl: FLConfig = field(default_factory=FLConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    experiment_name: str = "baseline1"
    save_dir: str = field(default_factory=lambda: str(
        Path(__file__).resolve().parent / "results"
    ))
