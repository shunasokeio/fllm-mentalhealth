# -*- coding: utf-8 -*-
save_path = "./FL_iid_qwen0.5b_newcluster_seed75"   ## !!!!change directory here
PARTITION_MODE = "iid" #mixed, cluster, hospital, iid, mixed_equalized
SEED = 75
COVERAGE_GAMMA = 0.0   # softmax temperature for coverage-aware aggregation; 0 = standard FedAvg
"""FL_train.py - Federated Learning for LLM Fine-tuning"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import copy
import math
import gc
import wandb
from collections import OrderedDict
from typing import Any, Callable, Dict, Tuple, Sequence
from dataclasses import dataclass

from datasets import load_dataset
import transformers
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    LlamaTokenizer
)
from torch.nn.utils.rnn import pad_sequence
import bitsandbytes as bnb

from peft import (
    PeftModel,
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training,
)
from peft.tuners.lora import LoraLayer

import numpy as np
import flwr as fl
from flwr.common.typing import NDArrays, Scalar
from flwr.common import Context
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, Partitioner, DirichletPartitioner
from flwr.client.mod import fixedclipping_mod
from datasets import Dataset, DatasetDict
import pandas as pd
from flwr.server.strategy import DifferentialPrivacyClientSideFixedClipping
from fllm.heterogeneity_logging import FedAvgWithHeterogeneityLog, FedAvgCoverageAware





IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"

import os
import glob
from types import SimpleNamespace

num_cpus = os.cpu_count()
print(f"Number of CPUs: {num_cpus}")
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")



# Configuration
cfg = SimpleNamespace(
    model=SimpleNamespace(
        name="Qwen/Qwen2.5-0.5B-Instruct",
        quantization=4,
        gradient_checkpointing=True,
        use_fast_tokenizer=False,
        lora=SimpleNamespace(
            peft_lora_r=32,  # Reduced from 64 to save memory
            peft_lora_alpha=16,
            target_modules="all-linear"
        )
    ),
    dataset=SimpleNamespace(
        name="ShenLab/MentalChat16K"
    ),
    flower=SimpleNamespace(
        num_clients=10,
        num_iid=0,
        num_pathological=0,
        client_resources=SimpleNamespace(
            num_cpus=num_cpus,
            num_gpus=num_gpus  # Each client needs full GPU access
        ),
        num_rounds=25,
        fraction_fit=0.3 if PARTITION_MODE == "cluster" else 0.5,
        fraction_evaluate=0.0,  # Disable federated evaluation to avoid OOM
        dp=SimpleNamespace(
            noise_mult=0.0,
            clip_norm=1.0
        )
    ),
    train=SimpleNamespace(
        training_arguments={
            "per_device_train_batch_size": 1,
            "logging_steps": 10,
            "max_steps": 20,  # Increased for proper training (5 clients × 200 steps × 100 rounds = 100K steps)
            "learning_rate": 0.0002,
            "output_dir": "./output",
            "optim": "paged_adamw_8bit",  # Use 8-bit optimizer for lower memory
            "gradient_accumulation_steps": 4,  # Reduced from 16 to save memory
            "weight_decay": 0.0,
            "max_grad_norm": 0.3,
            "gradient_checkpointing": True,
            "lr_scheduler_type": "constant",
            "warmup_ratio": 0.03,
            "group_by_length": False,
            "save_strategy": "no",
            "remove_unused_columns": False,
            "bf16": True,
            "fp16": False,
        },
        padding_side="right",
        evaluate_split=False,
        eval_test_size=0.02,  # Reserve 2% for global evaluation (faster testing)
        learning_rate_max=0.0002,
        learning_rate_min=0.0,
        save_every_round=1,
        source_max_len=1024,
        target_max_len=256,
        train_on_source=False,
    )
)

print("Configuration loaded:")

# Cluster labels path for FL clients (one client per cluster)
CLUSTER_LABELS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "embedded_data", "cluster_labels.npy")
CLUSTER_LABELS_PATH = os.path.normpath(CLUSTER_LABELS_PATH)
CLUSTERED_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "embedded_data", "clustered_dataset.csv")
CLUSTERED_CSV_PATH = os.path.normpath(CLUSTERED_CSV_PATH)


def get_train_test_indices(n_total: int, test_fraction: float = 0.02, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Return (train_indices, test_indices) using permutation + slice (same method everywhere)."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_total)
    test_size = int(round(n_total * test_fraction))
    train_indices = perm[:-test_size]
    test_indices = perm[-test_size:]
    return train_indices, test_indices


class ClusterPartitioner(Partitioner):
    """Partitioner that assigns each sample to a partition by cluster label (from cluster_labels.npy)."""

    def __init__(self, partition_ids: np.ndarray) -> None:
        super().__init__()
        self._partition_ids = partition_ids
        self._num_partitions = int(len(np.unique(partition_ids)))

    def load_partition(self, partition_id: int):
        indices = [i for i in range(len(self._partition_ids)) if self._partition_ids[i] == partition_id]
        subset = self.dataset.select(indices)
        if "cluster_id" not in subset.column_names:
            subset = subset.add_column("cluster_id", [partition_id] * len(indices))
        return subset

    @property
    def num_partitions(self) -> int:
        return self._num_partitions


class MixedPartitioner(Partitioner):
    """Partitioner that assigns data to IID, pathologically non-IID, and realistically non-IID clients (datasplitter-style IID + Dirichlet for path/realistic)."""

    def __init__(
        self,
        num_iid: int,
        num_pathological: int,
        num_realistic: int,
        alpha_pathological: float = 0.01,
        alpha_realistic: float = 0.5,
        partition_by: str = "cluster_id",
        seed: int = SEED,
    ) -> None:
        super().__init__()
        self._num_iid = num_iid
        self._num_path = num_pathological
        self._num_real = num_realistic
        self._alpha_path = alpha_pathological
        self._alpha_real = alpha_realistic
        self._partition_by = partition_by
        self._seed = seed
        self._num_partitions = num_iid + num_pathological + num_realistic
        self._partition_ids = None

    def _ensure_partition_ids(self) -> None:
        if self._partition_ids is not None:
            return
        n = len(self.dataset)
        labels = np.array(self.dataset[self._partition_by])
        rng = np.random.RandomState(self._seed)
        self._partition_ids = np.full(n, -1, dtype=np.int64)

        # Split indices into three pools (1/3 each)
        perm = rng.permutation(n)
        third = n // 3
        iid_pool = perm[:third]
        path_pool = perm[third : 2 * third]
        real_pool = perm[2 * third :]

        # IID clients (0 .. num_iid-1): per-label even split (datasplitter split_iid style)
        for label in np.unique(labels[iid_pool]):
            mask = labels[iid_pool] == label
            indices_for_label = iid_pool[mask]
            indices_for_label = rng.permutation(indices_for_label)
            shards = np.array_split(indices_for_label, self._num_iid)
            for client_idx, shard in enumerate(shards):
                self._partition_ids[shard] = client_idx

        # Pathological clients (num_iid .. num_iid+num_path-1): Dirichlet(alpha_path)
        path_client_offset = self._num_iid
        for label in np.unique(labels[path_pool]):
            mask = labels[path_pool] == label
            indices_for_label = path_pool[mask]
            rng.shuffle(indices_for_label)
            proportions = rng.dirichlet(self._alpha_path * np.ones(self._num_path))
            proportions = proportions / proportions.sum()
            cum = np.cumsum(np.round(proportions * len(indices_for_label)).astype(int))
            cum = np.minimum(cum, len(indices_for_label))
            cum[-1] = len(indices_for_label)
            splits = np.split(indices_for_label, cum[:-1])
            for client_idx, inds in enumerate(splits):
                if len(inds) > 0:
                    self._partition_ids[inds] = path_client_offset + client_idx

        # Realistic clients (num_iid+num_path .. total-1): Dirichlet(alpha_real)
        real_client_offset = self._num_iid + self._num_path
        for label in np.unique(labels[real_pool]):
            mask = labels[real_pool] == label
            indices_for_label = real_pool[mask]
            rng.shuffle(indices_for_label)
            proportions = rng.dirichlet(self._alpha_real * np.ones(self._num_real))
            proportions = proportions / proportions.sum()
            cum = np.cumsum(np.round(proportions * len(indices_for_label)).astype(int))
            cum = np.minimum(cum, len(indices_for_label))
            cum[-1] = len(indices_for_label)
            splits = np.split(indices_for_label, cum[:-1])
            for client_idx, inds in enumerate(splits):
                if len(inds) > 0:
                    self._partition_ids[inds] = real_client_offset + client_idx

        # Any unassigned (e.g. from rounding) assign to first client of corresponding group
        unassigned = np.where(self._partition_ids == -1)[0]
        for idx in unassigned:
            if idx in iid_pool:
                self._partition_ids[idx] = rng.randint(0, self._num_iid)
            elif idx in path_pool:
                self._partition_ids[idx] = path_client_offset
            else:
                self._partition_ids[idx] = real_client_offset

    def load_partition(self, partition_id: int):
        self._ensure_partition_ids()
        indices = [i for i in range(len(self._partition_ids)) if self._partition_ids[i] == partition_id]
        return self.dataset.select(indices)

    @property
    def num_partitions(self) -> int:
        return self._num_partitions


class EqualizedMixedPartitioner(MixedPartitioner):
    """MixedPartitioner with pool sizes proportional to number of clients.

    Standard MixedPartitioner splits training data into three equal 1/3
    pools regardless of how many clients draw from each pool.  With 2 IID
    clients and 4 path/hospital clients this gives IID clients ~2× more
    data per client than pathological or hospital clients, confounding
    distribution effects with volume effects.

    This variant sizes each pool as ``num_clients_in_pool × target`` where
    ``target = n_train // num_clients_total``, so every client receives
    the same number of samples drawn from its own pool.  Distribution
    character within each pool is unchanged: IID uses per-label even split,
    pathological uses Dirichlet(alpha_path), hospital uses
    Dirichlet(alpha_real).
    """

    def _ensure_partition_ids(self) -> None:
        if self._partition_ids is not None:
            return
        n = len(self.dataset)
        labels = np.array(self.dataset[self._partition_by])
        rng = np.random.RandomState(self._seed)
        self._partition_ids = np.full(n, -1, dtype=np.int64)

        # Proportional pool sizes: each client's target = n // num_clients
        target    = n // self._num_partitions
        iid_size  = self._num_iid  * target
        path_size = self._num_path * target
        real_size = self._num_real * target

        perm      = rng.permutation(n)
        iid_pool  = perm[:iid_size]
        path_pool = perm[iid_size : iid_size + path_size]
        real_pool = perm[iid_size + path_size : iid_size + path_size + real_size]

        print(f"[EqualizedMixedPartitioner] target_per_client = {n} // {self._num_partitions} = {target}")
        print(f"  IID  pool : {iid_size}  samples ({self._num_iid}  clients × {target})")
        print(f"  Path pool : {path_size} samples ({self._num_path} clients × {target})")
        print(f"  Real pool : {real_size} samples ({self._num_real} clients × {target})")
        remainder = n - iid_size - path_size - real_size
        if remainder:
            print(f"  Unused    : {remainder} samples (floor-division remainder)")

        # IID clients: per-label even split within iid_pool
        for label in np.unique(labels[iid_pool]):
            mask = labels[iid_pool] == label
            idx  = iid_pool[mask]
            idx  = rng.permutation(idx)
            for c, shard in enumerate(np.array_split(idx, self._num_iid)):
                self._partition_ids[shard] = c

        # Pathological clients: Dirichlet(alpha_path) per label within path_pool
        path_offset = self._num_iid
        for label in np.unique(labels[path_pool]):
            mask = labels[path_pool] == label
            idx  = path_pool[mask]
            rng.shuffle(idx)
            props = rng.dirichlet(self._alpha_path * np.ones(self._num_path))
            props = props / props.sum()
            cum   = np.cumsum(np.round(props * len(idx)).astype(int))
            cum   = np.minimum(cum, len(idx))
            cum[-1] = len(idx)
            for c, inds in enumerate(np.split(idx, cum[:-1])):
                if len(inds) > 0:
                    self._partition_ids[inds] = path_offset + c

        # Realistic/hospital clients: Dirichlet(alpha_real) per label within real_pool
        real_offset = self._num_iid + self._num_path
        for label in np.unique(labels[real_pool]):
            mask = labels[real_pool] == label
            idx  = real_pool[mask]
            rng.shuffle(idx)
            props = rng.dirichlet(self._alpha_real * np.ones(self._num_real))
            props = props / props.sum()
            cum   = np.cumsum(np.round(props * len(idx)).astype(int))
            cum   = np.minimum(cum, len(idx))
            cum[-1] = len(idx)
            for c, inds in enumerate(np.split(idx, cum[:-1])):
                if len(inds) > 0:
                    self._partition_ids[inds] = real_offset + c


class LabelIidPartitioner(Partitioner):
    """Strict IID partitioner: per-label even split across clients (datasplitter-style IID)."""

    def __init__(self, num_partitions: int, partition_by: str = "cluster_id", seed: int = SEED) -> None:
        super().__init__()
        self._num_partitions = num_partitions
        self._partition_by = partition_by
        self._seed = seed
        self._partition_ids: np.ndarray | None = None

    def _ensure_partition_ids(self) -> None:
        if self._partition_ids is not None:
            return
        n = len(self.dataset)
        labels = np.array(self.dataset[self._partition_by])
        rng = np.random.RandomState(self._seed)
        self._partition_ids = np.full(n, -1, dtype=np.int64)
        for label in np.unique(labels):
            mask = labels == label
            indices_for_label = np.where(mask)[0]
            indices_for_label = rng.permutation(indices_for_label)
            shards = np.array_split(indices_for_label, self._num_partitions)
            for client_idx, shard in enumerate(shards):
                if len(shard) > 0:
                    self._partition_ids[shard] = client_idx
        if np.any(self._partition_ids == -1):
            raise RuntimeError("LabelIidPartitioner: some samples were not assigned to any partition.")

    def load_partition(self, partition_id: int):
        self._ensure_partition_ids()
        indices = [i for i in range(len(self._partition_ids)) if self._partition_ids[i] == partition_id]
        return self.dataset.select(indices)

    @property
    def num_partitions(self) -> int:
        return self._num_partitions


def _make_cluster_preprocessor():
    """Load CSV, do train/test split (same method as hospital), filter train by non-noise, return (preprocessor, partition_ids, num_clients)."""
    if not os.path.isfile(CLUSTERED_CSV_PATH):
        raise FileNotFoundError(
            f"Clustered dataset CSV not found at {CLUSTERED_CSV_PATH}. "
            "Run clustering.py to create clustered_dataset.csv first."
        )
    if not os.path.isfile(CLUSTER_LABELS_PATH):
        raise FileNotFoundError(
            f"Cluster labels not found at {CLUSTER_LABELS_PATH}. "
            "Run clustering.py and save cluster_labels.npy first."
        )
    print(f"Loading clustered dataset from {CLUSTERED_CSV_PATH} ...")
    df = pd.read_csv(CLUSTERED_CSV_PATH)
    ds_full = Dataset.from_pandas(df, preserve_index=False)
    n_total = len(ds_full)
    cluster_labels = np.load(CLUSTER_LABELS_PATH)
    if len(cluster_labels) != n_total:
        raise ValueError(
            f"cluster_labels.npy length {len(cluster_labels)} does not match dataset size {n_total}."
        )
    train_indices, test_indices = get_train_test_indices(
        n_total, cfg.train.eval_test_size, SEED
    )
    train_mask = cluster_labels[train_indices] != -1
    train_indices = train_indices[train_mask]
    cluster_labels_train = cluster_labels[train_indices]
    unique_cids = sorted(set(cluster_labels_train))
    num_clients = len(unique_cids)
    cid_to_partition = {c: i for i, c in enumerate[Any](unique_cids)}
    partition_ids = np.array([cid_to_partition[c] for c in cluster_labels_train], dtype=np.int64)
    train_indices_all, _ = get_train_test_indices(n_total, cfg.train.eval_test_size, SEED)
    n_noise = int((cluster_labels[train_indices_all] == -1).sum())

    def preprocessor(_dataset_dict):
        train_ds = ds_full.select(train_indices.tolist())
        test_ds = ds_full.select(test_indices.tolist())
        return DatasetDict(train=train_ds, test=test_ds)

    return preprocessor, partition_ids, num_clients, n_noise


def _make_hospital_preprocessor() -> callable:
    """Preprocessor that builds train/test splits from clustered_dataset.csv."""

    def preprocessor(_dataset_dict):
        if not os.path.isfile(CLUSTERED_CSV_PATH):
            raise FileNotFoundError(
                f"Clustered dataset CSV not found at {CLUSTERED_CSV_PATH}. "
                "Run clustering.py to create clustered_dataset.csv first."
            )
        print(f"Loading clustered dataset from {CLUSTERED_CSV_PATH} ...")
        df = pd.read_csv(CLUSTERED_CSV_PATH)
        ds_full = Dataset.from_pandas(df, preserve_index=False)
        n_total = len(ds_full)
        train_indices, test_indices = get_train_test_indices(
            n_total, cfg.train.eval_test_size, SEED
        )
        train_ds = ds_full.select(train_indices.tolist())
        test_ds = ds_full.select(test_indices.tolist())
        print(f"Hospital mode: train={len(train_ds)} examples, test={len(test_ds)} examples")
        return DatasetDict({"train": train_ds, "test": test_ds})

    return preprocessor


def _visualize_partition(fds: FederatedDataset, train_partitioner: Partitioner, output_path: str, label_col: str = "cluster_id") -> None:
    """Plot cluster distribution per client (stacked bar). Requires partitions to have label_col (e.g. cluster_id)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("VISUALIZE_PARTITION: matplotlib not installed, skipping.", file=__import__("sys").stderr)
        return
    num_partitions = train_partitioner.num_partitions
    all_clusters = set()
    counts_per_client = []
    for pid in range(num_partitions):
        partition = fds.load_partition(pid, "train")
        if label_col not in partition.column_names:
            print(f"VISUALIZE_PARTITION: partition has no column '{label_col}', skipping plot.", file=__import__("sys").stderr)
            return
        labels = np.asarray(partition[label_col])
        unique, cts = np.unique(labels, return_counts=True)
        all_clusters.update(unique.tolist())
        counts_per_client.append(dict(zip(unique.tolist(), cts.tolist())))
    clusters_sorted = sorted(all_clusters)
    n_clusters = len(clusters_sorted)
    cluster_to_idx = {c: i for i, c in enumerate(clusters_sorted)}
    matrix = np.zeros((num_partitions, n_clusters), dtype=np.int64)
    for pid, cts in enumerate(counts_per_client):
        for c, count in cts.items():
            matrix[pid, cluster_to_idx[c]] = count

    # Client counts
    client_totals = matrix.sum(axis=1)

    # Global distribution as fractions, then scaled so total height ~= average client size
    global_counts = matrix.sum(axis=0)
    global_total = float(global_counts.sum()) if global_counts.sum() > 0 else 1.0
    global_frac = global_counts / global_total
    avg_client_total = float(client_totals.mean()) if client_totals.size > 0 else 1.0
    global_scaled = global_frac * avg_client_total

    num_bars = num_partitions + 1  # one bar for global + one per client
    fig, ax = plt.subplots(figsize=(max(8, num_bars * 1.2), 6))
    x = np.arange(num_bars)
    width = 0.75
    bottom = np.zeros(num_bars)
    colors = plt.cm.tab20(np.linspace(0, 1, n_clusters)) if n_clusters <= 20 else plt.cm.tab20b(np.linspace(0, 1, n_clusters))

    for i, c in enumerate(clusters_sorted):
        heights = np.zeros(num_bars, dtype=float)
        # Global bar at index 0 (proportions scaled to avg client height)
        heights[0] = global_scaled[i]
        # Client counts at indices 1..num_partitions
        heights[1:] = matrix[:, i]
        if heights.sum() > 0:
            ax.bar(x, heights, width, bottom=bottom, label=f"Cluster {c}", color=colors[i % len(colors)])
            bottom = bottom + heights

    ax.set_xlabel("Global (left) and client IDs")
    ax.set_ylabel("Number of samples (clients); global bar scaled")
    ax.set_title(f"Per-client cluster counts vs global distribution ({PARTITION_MODE} mode)")
    tick_labels = ["global"] + [str(i) for i in range(num_partitions)]
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.legend(loc="upper right", bbox_to_anchor=(1.15, 1), ncol=1, fontsize=8)

    if PARTITION_MODE in ("mixed", "mixed_equalized") and hasattr(train_partitioner, "_num_iid"):
        n_iid = train_partitioner._num_iid
        n_path = train_partitioner._num_path
        y_offset = max(1.0, bottom[1:].max() * 0.02) if bottom[1:].size > 0 else 1.0
        for client_idx in range(num_partitions):
            bar_x = client_idx + 1  # clients start at index 1
            kind = "IID" if client_idx < n_iid else ("path" if client_idx < n_iid + n_path else "real")
            ax.annotate(kind, (bar_x, bottom[bar_x] + y_offset), ha="center", va="bottom", fontsize=7, rotation=90)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Partition visualization saved: {output_path}")


print(f"Partition mode: {PARTITION_MODE}")

if PARTITION_MODE == "cluster":
    # Load CSV, do train/test split (same method as hospital), filter train by non-noise
    print("Cluster mode: loading CSV and reserving evaluation set...")
    cluster_preprocessor, partition_ids, num_clients, n_noise = _make_cluster_preprocessor()
    cfg.flower.num_clients = num_clients
    print(f"Using cluster-based partitioning: {num_clients} clients (one per cluster); ignored {n_noise} noise points")

    train_partitioner: Partitioner = ClusterPartitioner(partition_ids)
    fds = FederatedDataset(
        dataset=cfg.dataset.name,
        shuffle=False,
        preprocessor=cluster_preprocessor,
        partitioners={"train": train_partitioner},
    )

    # Eval dataset is the "test" split from the preprocessor
    global_eval_dataset = fds.load_split("test")
    print(f"Global evaluation dataset has {len(global_eval_dataset)} examples")
    # Print length of each partition
    print(f"\nPartition sizes (total partitions: {train_partitioner.num_partitions}):")
    for pid in range(train_partitioner.num_partitions):
        partition = fds.load_partition(pid, "train")
        print(f"  Partition {pid}: {len(partition)} examples")
    total_partitioned = sum(len(fds.load_partition(pid, "train")) for pid in range(train_partitioner.num_partitions))
    print(f"Total examples across all partitions: {total_partitioned}")
    print(f"Expected (non-noise train): {len(partition_ids)}")

elif PARTITION_MODE == "hospital":
    # 10 hospitals with different cluster distributions using DirichletPartitioner
    num_hospitals = 10
    cfg.flower.num_clients = num_hospitals
    hospital_preprocessor = _make_hospital_preprocessor()

    # DirichletPartitioner over the 'cluster_id' column from clustered_dataset.csv
    train_partitioner = DirichletPartitioner(
        num_partitions=num_hospitals,
        partition_by="cluster_id",
        alpha=0.5,           # smaller => more non-IID across hospitals
        min_partition_size=10,
    )

    fds = FederatedDataset(
        dataset=cfg.dataset.name,   # actual data comes from preprocessor's DatasetDict
        shuffle=False,
        preprocessor=hospital_preprocessor,
        partitioners={"train": train_partitioner},
    )

    global_eval_dataset = fds.load_split("test")
    print(f"Hospital mode: global evaluation dataset has {len(global_eval_dataset)} examples")
    print(f"\nHospital partition sizes (total hospitals: {num_hospitals}):")
    for pid in range(num_hospitals):
        partition = fds.load_partition(pid, "train")
        print(f"  Hospital {pid}: {len(partition)} examples")
    total_partitioned = sum(len(fds.load_partition(pid, "train")) for pid in range(num_hospitals))
    print(f"Total examples across all hospital partitions: {total_partitioned}")

elif PARTITION_MODE == "iid":
    num_clients = int(os.environ.get("IID_NUM_CLIENTS", str(cfg.flower.num_clients)))
    cfg.flower.num_clients = num_clients
    hospital_preprocessor = _make_hospital_preprocessor()
    train_partitioner = LabelIidPartitioner(
        num_partitions=num_clients,
        partition_by="cluster_id",
        seed=SEED,
    )
    fds = FederatedDataset(
        dataset=cfg.dataset.name,
        shuffle=False,
        preprocessor=hospital_preprocessor,
        partitioners={"train": train_partitioner},
    )
    global_eval_dataset = fds.load_split("test")
    print(f"IID mode: global evaluation dataset has {len(global_eval_dataset)} examples")
    print(f"\nIID partition sizes (total clients: {train_partitioner.num_partitions}):")
    for pid in range(train_partitioner.num_partitions):
        partition = fds.load_partition(pid, "train")
        print(f"  Client {pid}: {len(partition)} examples")
    total_partitioned = sum(len(fds.load_partition(pid, "train")) for pid in range(train_partitioner.num_partitions))
    print(f"Total examples across all IID partitions: {total_partitioned}")

elif PARTITION_MODE == "mixed":
    num_iid = int(os.environ.get("MIXED_NUM_IID", "2"))
    num_pathological = int(os.environ.get("MIXED_NUM_PATHOLOGICAL", "4"))
    num_realistic = int(os.environ.get("MIXED_NUM_REALISTIC", "4"))
    alpha_pathological = float(os.environ.get("MIXED_ALPHA_PATH", "0.01"))
    alpha_realistic = float(os.environ.get("MIXED_ALPHA_REAL", "0.5"))
    cfg.flower.num_clients = num_iid + num_pathological + num_realistic
    cfg.flower.num_iid = num_iid
    cfg.flower.num_pathological = num_pathological
    hospital_preprocessor = _make_hospital_preprocessor()
    train_partitioner = MixedPartitioner(
        num_iid=num_iid,
        num_pathological=num_pathological,
        num_realistic=num_realistic,
        alpha_pathological=alpha_pathological,
        alpha_realistic=alpha_realistic,
        partition_by="cluster_id",
        seed=SEED,
    )
    fds = FederatedDataset(
        dataset=cfg.dataset.name,
        shuffle=False,
        preprocessor=hospital_preprocessor,
        partitioners={"train": train_partitioner},
    )
    global_eval_dataset = fds.load_split("test")
    print(f"Mixed mode: global evaluation dataset has {len(global_eval_dataset)} examples")
    print(f"Mixed clients: IID 0..{num_iid - 1}, pathological {num_iid}..{num_iid + num_pathological - 1}, realistic {num_iid + num_pathological}..{cfg.flower.num_clients - 1}")
    print(f"\nMixed partition sizes (total: {train_partitioner.num_partitions}):")
    for pid in range(train_partitioner.num_partitions):
        partition = fds.load_partition(pid, "train")
        print(f"  Partition {pid}: {len(partition)} examples")
    total_partitioned = sum(len(fds.load_partition(pid, "train")) for pid in range(train_partitioner.num_partitions))
    print(f"Total examples across all partitions: {total_partitioned}")

elif PARTITION_MODE == "mixed_equalized":
    num_iid = int(os.environ.get("MIXED_NUM_IID", "2"))
    num_pathological = int(os.environ.get("MIXED_NUM_PATHOLOGICAL", "4"))
    num_realistic = int(os.environ.get("MIXED_NUM_REALISTIC", "4"))
    alpha_pathological = float(os.environ.get("MIXED_ALPHA_PATH", "0.01"))
    alpha_realistic = float(os.environ.get("MIXED_ALPHA_REAL", "0.5"))
    cfg.flower.num_clients = num_iid + num_pathological + num_realistic
    hospital_preprocessor = _make_hospital_preprocessor()
    train_partitioner = EqualizedMixedPartitioner(
        num_iid=num_iid,
        num_pathological=num_pathological,
        num_realistic=num_realistic,
        alpha_pathological=alpha_pathological,
        alpha_realistic=alpha_realistic,
        partition_by="cluster_id",
        seed=SEED,
    )
    fds = FederatedDataset(
        dataset=cfg.dataset.name,
        shuffle=False,
        preprocessor=hospital_preprocessor,
        partitioners={"train": train_partitioner},
    )
    global_eval_dataset = fds.load_split("test")
    print(f"Mixed-equalized mode: global evaluation dataset has {len(global_eval_dataset)} examples")
    print(f"Clients: IID 0..{num_iid-1}, pathological {num_iid}..{num_iid+num_pathological-1}, "
          f"realistic {num_iid+num_pathological}..{cfg.flower.num_clients-1}")
    print(f"\nMixed-equalized partition sizes (total: {train_partitioner.num_partitions}):")
    for pid in range(train_partitioner.num_partitions):
        partition = fds.load_partition(pid, "train")
        print(f"  Partition {pid}: {len(partition)} examples")

else:
    raise ValueError(f"Unknown PARTITION_MODE '{PARTITION_MODE}'. "
                     f"Use 'cluster', 'hospital', 'mixed', 'mixed_equalized', or 'iid'.")


_viz_path = f"{save_path}_distribution.png"
_visualize_partition(fds, train_partitioner, _viz_path)


def find_all_linear_names(model, bits=4):
    """Find all linear layer names for LoRA target modules."""
    cls = bnb.nn.Linear4bit if bits == 4 else (bnb.nn.Linear8bitLt if bits == 8 else torch.nn.Linear)
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    
    if 'lm_head' in lora_module_names:
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    
    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg

def get_model(model_cfg: SimpleNamespace):
    """Load model with appropriate quantization config and other optimizations."""
    use_cuda = torch.cuda.is_available()
    quantization_config = None
    model_name = model_cfg.name
    
    compute_dtype = torch.bfloat16 if use_cuda else torch.float32
    
    if use_cuda:
        if model_cfg.quantization == 4:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        elif model_cfg.quantization == 8:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
            )
        else:
            raise ValueError(
                f"Use 4-bit or 8-bit quantization. You passed: {model_cfg.quantization}"
            )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if use_cuda else None,
    )
    
    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    if use_cuda:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )

    # Determine target modules
    target_modules = model_cfg.lora.target_modules
    if target_modules == "all-linear":
        target_modules = find_all_linear_names(model, bits=model_cfg.quantization)
        print(f"Auto-detected LoRA target modules: {target_modules}")
    elif isinstance(target_modules, str):
        target_modules = [target_modules]
    
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        lora_dropout=0.075,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        bias="none",
    )

    peft_model = get_peft_model(model, peft_config)
    if not use_cuda:
        peft_model.enable_input_require_grads()

    # Apply dtype adjustments
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(compute_dtype)
        if 'norm' in name:
            module = module.to(torch.float32)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if module.weight.dtype == torch.float32:
                    module = module.to(compute_dtype)

    if model_cfg.gradient_checkpointing:
        peft_model.config.use_cache = False

    return peft_model

@dataclass
class DataCollatorForCausalLM(object):
    """Data collator for causal language modeling."""
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool = False

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        sources = []
        targets = []
        use_chat_template = hasattr(self.tokenizer, "apply_chat_template")
        for idx, example in enumerate(instances):
            # Attempt to build a multi-turn chat if possible
            messages = []
            # Multi-turn: if example has a 'messages' field (list of dicts with 'role' and 'content')
            if 'messages' in example and isinstance(example['messages'], list):
                for msg in example['messages']:
                    if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                        messages.append({"role": msg["role"], "content": str(msg["content"])})
            else:
                # Single-turn fallback: try to extract user/assistant turns
                if 'instruction' in example:
                    user_input = (example.get('input', '') if isinstance(example.get('input', ''), str) else str(example.get('input', '') or '')).strip()
                    target = (example.get('output', '') if isinstance(example.get('output', ''), str) else str(example.get('output', '') or '')).strip()
                elif 'Context' in example and 'Response' in example:
                    user_input = str(example['Context'])
                    target = str(example['Response'])
                elif 'input' in example and 'output' in example:
                    user_input = str(example['input'])
                    target = str(example['output'])
                else:
                    vals = list(example.values())
                    user_input = str(vals[0]) if len(vals) > 0 else ''
                    target = str(vals[1]) if len(vals) > 1 else ''
                messages = [{"role": "user", "content": user_input}]
            # Always use chat template if available
            if use_chat_template:
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                sources.append(prompt)
                # For single-turn, target is just the assistant's response
                if len(messages) == 1 and 'target' in locals():
                    targets.append(f"{target}{self.tokenizer.eos_token}")
                else:
                    # For multi-turn, try to extract the last assistant message as target
                    last_assistant = next((m['content'] for m in reversed(messages) if m['role'] == 'assistant'), None)
                    if last_assistant is not None:
                        targets.append(f"{last_assistant}{self.tokenizer.eos_token}")
                    else:
                        # Fallback: empty target
                        targets.append(f"{self.tokenizer.eos_token}")
            else:
                # Fallback to old prompt style
                sources.append(f"{self.tokenizer.bos_token}User: {user_input}\nAssistant:")
                targets.append(f"{target}{self.tokenizer.eos_token}")
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )

        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))

        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None

        # Debug: Print first example formatting (only once)
        if not hasattr(self, '_debug_printed'):
            self._debug_printed = True
            print("\n" + "="*80)
            print("DATA COLLATOR DEBUG - First Training Example:")
            print("="*80)
            full_text = self.tokenizer.decode(input_ids[0])
            print(f"\nFULL INPUT TEXT:\n{full_text}\n")
            if labels is not None:
                # Find where labels start (first non-ignored token)
                first_real_label_idx = (labels[0] != IGNORE_INDEX).nonzero()
                if len(first_real_label_idx) > 0:
                    start_idx = first_real_label_idx[0].item()
                    label_tokens = input_ids[0][start_idx:]
                    label_text = self.tokenizer.decode(label_tokens)
                    print(f"TRAINING TARGET (what model learns to generate):\n{label_text}\n")
                    print(f"SOURCE LENGTH: {start_idx} tokens (masked from loss)")
                    print(f"TARGET LENGTH: {len(label_tokens)} tokens (trained)")
            print("="*80 + "\n")

        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def get_tokenizer_and_data_collator(
    model_name: str, 
    train_cfg: SimpleNamespace,
    use_fast: bool = False, 
    padding_side: str = "right"
):
    """Initialize tokenizer and data collator."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=use_fast, 
        padding_side=padding_side
    )
    
    # Handle missing pad token
    if getattr(tokenizer, 'pad_token', None) is None or tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({'pad_token': DEFAULT_PAD_TOKEN})
    
    # Handle LLaMA tokenizers
    if 'llama' in model_name.lower() or isinstance(tokenizer, LlamaTokenizer):
        print('Adding special tokens for LLaMA.')
        tokenizer.add_special_tokens({
            "eos_token": tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id) if tokenizer.eos_token_id else "</s>",
            "bos_token": tokenizer.convert_ids_to_tokens(tokenizer.bos_token_id) if tokenizer.bos_token_id else "<s>",
            "unk_token": tokenizer.convert_ids_to_tokens(0) if hasattr(tokenizer, 'convert_ids_to_tokens') else "<unk>",
        })
    
    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        source_max_len=train_cfg.source_max_len,
        target_max_len=train_cfg.target_max_len,
        train_on_source=train_cfg.train_on_source,
        predict_with_generate=False,
    )
    
    return tokenizer, data_collator

tokenizer, data_collator = get_tokenizer_and_data_collator(
    cfg.model.name,
    cfg.train,
    cfg.model.use_fast_tokenizer,
    cfg.train.padding_side,
)

#i've understood to this point Nov 24 16:01

def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)

def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""

    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))

class FlowerClient(
    fl.client.NumPyClient
):  # pylint: disable=too-many-instance-attributes
    """Standard Flower client for CNN training."""

    def __init__(
        self,
        model_cfg: SimpleNamespace,
        train_cfg: SimpleNamespace,
        trainset,
        tokenizer,
        data_collator,
        save_path,
        partition_id: int = -1,
    ):  # pylint: disable=too-many-arguments
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.train_cfg = train_cfg
        self.training_arguments = Seq2SeqTrainingArguments(**train_cfg.training_arguments)
        self.tokenizer = tokenizer
        self.data_collator = data_collator
        self.save_path = save_path
        self.partition_id = partition_id

        # Instantiate model
        self.model = get_model(model_cfg)
        trainable, all_parameters = self.model.get_nb_trainable_parameters()
        print(f"Trainable parameters: {trainable}")
        print(f"All parameters: {all_parameters}")
        print(f"Trainable (%): {100*trainable / all_parameters:.3f}")

        self.trainset = trainset

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the parameters of the current net."""

        state_dict = get_peft_model_state_dict(self.model)
        # Cast to float32 to avoid unsupported bfloat16 numpy conversion errors
        return [val.detach().to(torch.float32).cpu().numpy() for _, val in state_dict.items()]

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Implement distributed fit function for a given client."""
        # Clear GPU memory before starting
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        set_parameters(self.model, parameters)

        total_rounds = config.get("total_rounds", cfg.flower.num_rounds)
        new_lr = cosine_annealing(
            int(config["current_round"]),
            total_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )

        self.training_arguments.learning_rate = new_lr
        self.training_arguments.output_dir = self.save_path

        evalset = None
        if self.train_cfg.evaluate_split:
            n = len(self.trainset)
            train_indices, test_indices = get_train_test_indices(n, test_fraction=0.2, seed=SEED)
            trainset = self.trainset.select(train_indices.tolist())
            evalset = self.trainset.select(test_indices.tolist())
        else:
            trainset = self.trainset

        # Use Seq2SeqTrainer like in cent.py
        trainer = Seq2SeqTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            train_dataset=trainset,
            eval_dataset=evalset,
            data_collator=self.data_collator,
        )

        metrics = {}
        if self.train_cfg.evaluate_split:
            eval_res = trainer.evaluate()
            metrics['eval_loss'] = eval_res['eval_loss']
            print(eval_res)

        # Do local training
        results = trainer.train()

        metrics = {**metrics, "train_loss": results.training_loss, "partition_id": self.partition_id}

        # Aggressive GPU memory cleanup after training
        del trainer, results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Wait for all operations to complete
            torch.cuda.ipc_collect()  # Clean up inter-process CUDA memory
        gc.collect()  # Force garbage collection

        return (
            self.get_parameters({}),
            len(self.trainset),
            metrics,
        )

def gen_client_fn(
    fds,
    tokenizer,
    data_collator,
    model_cfg: SimpleNamespace,
    train_cfg: SimpleNamespace,
    save_path: str,
) -> Callable[[str], FlowerClient]:  # pylint: disable=too-many-arguments
    """Generate the client function that creates the Flower Clients."""

    def client_fn(context: Context) -> FlowerClient:
        """Create a Flower client representing a single organization."""
        # Let's get the partition corresponding to the i-th client
        partition_id = int(context.node_config["partition-id"])
        client_trainset = fds.load_partition(partition_id, "train")
        return FlowerClient(
            model_cfg,
            train_cfg,
            client_trainset,
            tokenizer,
            data_collator,
            save_path,
            partition_id=partition_id,
        ).to_client()

    return client_fn

client = fl.client.ClientApp(
    client_fn=gen_client_fn(
        fds,
        tokenizer,
        data_collator,
        cfg.model,
        cfg.train,
        save_path,
    ),
    mods=[fixedclipping_mod]
)

################################ server components #############################


# Get function that will be executed by the strategy's evaluate() method
# Here we use it to save global model checkpoints
def get_evaluate_fn(model_cfg, train_cfg, eval_dataset, tokenizer, data_collator, save_every_round, total_round, save_path, starting_round=0):
    """Return an evaluation function for federated evaluation and model saving."""

    def evaluate(server_round: int, parameters, config):
        # Calculate cumulative round number
        cumulative_round = starting_round + server_round
        
        # Save model checkpoint
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            model = get_model(model_cfg)
            set_parameters(model, parameters)
            model.save_pretrained(f"{save_path}/peft_{cumulative_round}")
            print(f"\n{'='*60}")
            print(f"  CHECKPOINT SAVED")
            print(f"  Current Flower Round: {server_round}")
            print(f"  Total Cumulative Round: {cumulative_round}")
            print(f"{'='*60}\n")
            
            # Clean up model immediately after saving
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()
        
        # Evaluation disabled - return immediately to avoid memory allocation
        return 0.0, {}

    return evaluate


# Get a function that will be used to construct the config that the client's
# fit() method will receive
def get_on_fit_config():
    def fit_config_fn(server_round: int):
        fit_config = {
            "current_round": server_round,
            "total_rounds": cfg.flower.num_rounds,
        }
        return fit_config

    return fit_config_fn


def make_fit_metrics_fn(save_path: str, starting_round: int = 0):
    """Factory returning a fit_metrics_aggregation_fn that logs per-round loss to a JSONL file."""
    import json as _json
    _round = [starting_round]

    def fit_weighted_average(metrics):
        _round[0] += 1
        losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        weighted_avg_loss = sum(losses) / sum(examples)

        print("\n" + "="*60)
        print(f"  FEDERATED TRAINING METRICS")
        print("="*60)
        print(f"  Round: {_round[0]}")
        print(f"  Weighted Average Loss: {weighted_avg_loss:.6f}")
        print(f"  Total Examples: {sum(examples)}")
        print(f"  Participating Clients: {len(metrics)}")
        print("="*60 + "\n")

        # Append to JSONL loss curve file for later visualization
        entry = {"round": _round[0], "loss": weighted_avg_loss}
        with open(f"{save_path}_loss_curve.jsonl", "a") as _f:
            _f.write(_json.dumps(entry) + "\n")

        if wandb.run is not None:
            wandb.log({"train_loss_weighted_avg": weighted_avg_loss, "round": _round[0]})

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return {"train_loss": weighted_avg_loss}

    return fit_weighted_average

def get_latest_checkpoint(save_path):
    """Find the latest checkpoint in save_path. Returns (checkpoint_path, round_number) or (None, 0)."""
    checkpoints = glob.glob(f"{save_path}/peft_*")
    if not checkpoints:
        return None, 0
    
    # Extract round numbers and find the latest
    checkpoint_rounds = []
    for cp in checkpoints:
        try:
            round_num = int(cp.split('peft_')[-1])
            checkpoint_rounds.append((round_num, cp))
        except ValueError:
            continue
    
    if checkpoint_rounds:
        latest_checkpoint = max(checkpoint_rounds, key=lambda x: x[0])
        return latest_checkpoint[1], latest_checkpoint[0]  # (path, round_num)
    return None, 0

def server_fn(context: Context):
    # Initialize wandb for server-side logging
    wandb.init(
        project="federated-llm-training",
        name=f"server_{cfg.model.name.split('/')[-1]}",
        config={
            "model": cfg.model.name,
            "num_clients": cfg.flower.num_clients,
            "num_rounds": cfg.flower.num_rounds,
            "fraction_fit": cfg.flower.fraction_fit,
            "learning_rate_max": cfg.train.learning_rate_max,
        }
    )

    # Check for existing checkpoint to resume from
    latest_checkpoint, starting_round = get_latest_checkpoint(save_path)
    initial_parameters = None
    
    if latest_checkpoint:
        print(f"\n{'='*60}")
        print(f"  RESUMING FROM CHECKPOINT: {latest_checkpoint}")
        print(f"  Starting Round: {starting_round}")
        print(f"  Will train for {cfg.flower.num_rounds} more rounds")
        print(f"  Final round will be: {starting_round + cfg.flower.num_rounds}")
        print(f"{'='*60}\n")
        
        # Load the checkpoint on CPU to avoid OOM
        # Temporarily disable CUDA for loading checkpoint
        original_device = torch.cuda.is_available()
        
        # Create a minimal model config for CPU loading
        cpu_model_cfg = copy.deepcopy(cfg.model)
        
        # Load base model on CPU
        base_model = AutoModelForCausalLM.from_pretrained(
            cpu_model_cfg.name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            device_map="cpu",
        )
        
        from peft import PeftModel
        model = PeftModel.from_pretrained(base_model, latest_checkpoint, device_map="cpu")
        
        # Get parameters as NDArrays
        state_dict = get_peft_model_state_dict(model)
        initial_parameters = [val.detach().to(torch.float32).cpu().numpy() for _, val in state_dict.items()]
        
        # Clean up
        del model, base_model
        torch.cuda.empty_cache()
        print("Checkpoint loaded successfully!\n")
    else:
        print("\nNo existing checkpoint found. Starting from scratch.\n")

    # Define the Strategy
    strategy = FedAvgCoverageAware(
        gamma=COVERAGE_GAMMA,
        min_available_clients=cfg.flower.num_clients, # total clients
        fraction_fit=cfg.flower.fraction_fit, # ratio of clients to sample
        fraction_evaluate=0.0, # Disabled to avoid OOM during training
        initial_parameters=fl.common.ndarrays_to_parameters(initial_parameters) if initial_parameters else None,
        # A (optional) function used to configure a "fit()" round
        on_fit_config_fn=get_on_fit_config(),
        # A (optional) function to aggregate metrics sent by clients
        fit_metrics_aggregation_fn=make_fit_metrics_fn(save_path, starting_round),
        # Performs federated evaluation and saves the global model.
        evaluate_fn=get_evaluate_fn(
            cfg.model,
            cfg.train,
            global_eval_dataset,
            tokenizer,
            data_collator,
            cfg.train.save_every_round,
            cfg.flower.num_rounds,
            save_path,
            starting_round=starting_round
        ),
        log_path="heterogeneity_labelled_log.csv",
        lora_rank=cfg.model.lora.peft_lora_r,
        num_iid=cfg.flower.num_iid,
        num_pathological=cfg.flower.num_pathological,
        starting_round=starting_round,
        ema_path=f"{save_path}_global_update_ema.npy",
    )

    # Add Differential Privacy
    sampled_clients = cfg.flower.num_clients*strategy.fraction_fit
    strategy = DifferentialPrivacyClientSideFixedClipping(
        strategy,
        noise_multiplier=cfg.flower.dp.noise_mult,
        clipping_norm=cfg.flower.dp.clip_norm,
        num_sampled_clients=sampled_clients
    )

    # Number of rounds to run the simulation
    num_rounds = cfg.flower.num_rounds
    config = fl.server.ServerConfig(num_rounds=num_rounds)

    return fl.server.ServerAppComponents(strategy=strategy, config=config)

server = fl.server.ServerApp(server_fn=server_fn)

if __name__ == "__main__":
    from logging import ERROR
    import ray
    
    # Set PyTorch CUDA memory allocator to reduce fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print("Starting Federated Learning simulation...")
    # Convert SimpleNamespace to dict to avoid TypeError
    client_resources = vars(cfg.flower.client_resources)
    backend_setup = {"logging_level": ERROR, "log_to_driver": False}
    
    fl.simulation.run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=cfg.flower.num_clients,
        backend_config={
            "client_resources": client_resources,
            "init_args": backend_setup
        }
    )
    
    print("Federated Learning simulation completed!")
    print(f"Final model saved to: {save_path}")