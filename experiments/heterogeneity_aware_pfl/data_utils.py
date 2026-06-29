"""Data loading, two-alpha Dirichlet partitioning, and per-client train/val/test splits."""

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from datasets import Dataset

from .config import CLUSTERED_CSV_PATH, CLUSTER_LABELS_PATH, ExperimentConfig


def load_clustered_dataset() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load the clustered MentalChat16K CSV and cluster labels."""
    df = pd.read_csv(CLUSTERED_CSV_PATH)
    labels = np.load(CLUSTER_LABELS_PATH)
    # Filter out noise points (cluster_id == -1) if any
    valid_mask = labels >= 0
    df = df[valid_mask].reset_index(drop=True)
    labels = labels[valid_mask]
    return df, labels


def create_two_alpha_partitioner(
    labels: np.ndarray,
    num_iid: int,
    num_noniid: int,
    alpha_iid: float,
    alpha_noniid: float,
    seed: int,
) -> Dict[int, np.ndarray]:
    """Partition data indices into clients using two-pool Dirichlet allocation.

    First splits each cluster's data into two equal pools (IID pool, non-IID pool),
    then applies Dirichlet(alpha_iid) within the IID pool across IID clients, and
    Dirichlet(alpha_noniid) within the non-IID pool across non-IID clients.

    Clients 0..(num_iid-1) are IID.
    Clients num_iid..(num_iid+num_noniid-1) are non-IID.

    Returns: {client_id: array_of_global_indices}
    """
    rng = np.random.RandomState(seed)
    num_clients = num_iid + num_noniid
    unique_labels = np.unique(labels)

    client_indices: Dict[int, list] = {i: [] for i in range(num_clients)}

    # Target samples per client for balanced pool sizes
    n_total = len(labels)
    target_per_client = n_total / num_clients

    for label in unique_labels:
        label_indices = np.where(labels == label)[0]
        rng.shuffle(label_indices)
        n_label = len(label_indices)

        # Split label's data proportionally: num_iid clients vs num_noniid clients
        n_iid_pool = int(round(n_label * num_iid / num_clients))
        iid_pool = label_indices[:n_iid_pool]
        noniid_pool = label_indices[n_iid_pool:]

        # IID pool: Dirichlet(alpha_iid) across IID clients
        if len(iid_pool) > 0 and num_iid > 0:
            props_iid = rng.dirichlet([alpha_iid] * num_iid)
            props_iid = props_iid / props_iid.sum()
            cum = np.cumsum(props_iid)
            splits = (cum[:-1] * len(iid_pool)).astype(int)
            split_parts = np.split(iid_pool, splits)
            for i, part in enumerate(split_parts):
                client_indices[i].extend(part.tolist())

        # Non-IID pool: Dirichlet(alpha_noniid) across non-IID clients
        if len(noniid_pool) > 0 and num_noniid > 0:
            props_noniid = rng.dirichlet([alpha_noniid] * num_noniid)
            props_noniid = props_noniid / props_noniid.sum()
            cum = np.cumsum(props_noniid)
            splits = (cum[:-1] * len(noniid_pool)).astype(int)
            split_parts = np.split(noniid_pool, splits)
            for j, part in enumerate(split_parts):
                client_indices[num_iid + j].extend(part.tolist())

    # Convert to numpy arrays and shuffle
    for client_id in client_indices:
        arr = np.array(client_indices[client_id])
        rng.shuffle(arr)
        client_indices[client_id] = arr

    return client_indices


def split_client_data(
    indices: np.ndarray,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split a single client's indices into train/val/test sets."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(indices)
    n = len(perm)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    n_train = n - n_val - n_test

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    return train_idx, val_idx, test_idx


def prepare_all_clients(
    config: ExperimentConfig,
) -> Dict[int, Dict[str, Dataset]]:
    """Prepare all client datasets.

    Returns: {client_id: {"train": Dataset, "val": Dataset, "test": Dataset, "type": str}}
    """
    df, labels = load_clustered_dataset()
    fl = config.fl

    client_indices = create_two_alpha_partitioner(
        labels=labels,
        num_iid=fl.num_iid_clients,
        num_noniid=fl.num_noniid_clients,
        alpha_iid=fl.alpha_iid,
        alpha_noniid=fl.alpha_noniid,
        seed=fl.seed,
    )

    client_data = {}
    for client_id, indices in client_indices.items():
        train_idx, val_idx, test_idx = split_client_data(
            indices, fl.train_fraction, fl.val_fraction, fl.test_fraction,
            seed=fl.seed + client_id,
        )

        client_type = "iid" if client_id < fl.num_iid_clients else "noniid"

        client_data[client_id] = {
            "train": Dataset.from_pandas(df.iloc[train_idx].reset_index(drop=True)),
            "val": Dataset.from_pandas(df.iloc[val_idx].reset_index(drop=True)),
            "test": Dataset.from_pandas(df.iloc[test_idx].reset_index(drop=True)),
            "type": client_type,
        }

    # Print distribution summary
    print("\n" + "=" * 60)
    print("Client Data Distribution Summary")
    print("=" * 60)
    for cid in sorted(client_data.keys()):
        cd = client_data[cid]
        ctype = cd["type"]
        n_train = len(cd["train"])
        n_val = len(cd["val"])
        n_test = len(cd["test"])
        print(f"  Client {cid:2d} ({ctype:6s}): train={n_train:5d}, val={n_val:4d}, test={n_test:4d}")

        # Show cluster distribution for this client
        train_df = cd["train"].to_pandas()
        if "cluster_id" in train_df.columns:
            dist = train_df["cluster_id"].value_counts().sort_index()
            top = dist.head(5).to_dict()
            print(f"           top clusters: {top}")
    print("=" * 60 + "\n")

    return client_data


def plot_client_distribution(
    client_data: Dict[int, Dict],
    output_path: Path,
    label_col: str = "cluster_id",
    alpha_iid: float = 100.0,
    alpha_noniid: float = 0.01,
) -> None:
    """Plot cluster distribution per client as a stacked bar chart.

    Matches the style of the existing _distribution.png files.
    First bar shows global distribution, remaining bars show per-client.
    IID and non-IID clients are visually grouped with alpha values displayed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping distribution plot.")
        return

    # Build counts matrix
    all_clusters: set = set()
    counts_per_client = {}
    for cid in sorted(client_data.keys()):
        train_df = client_data[cid]["train"].to_pandas()
        if label_col not in train_df.columns:
            print(f"Column '{label_col}' not found, skipping distribution plot.")
            return
        labels = train_df[label_col].values
        unique, cts = np.unique(labels, return_counts=True)
        all_clusters.update(unique.tolist())
        counts_per_client[cid] = dict(zip(unique.tolist(), cts.tolist()))

    client_ids = sorted(client_data.keys())
    clusters_sorted = sorted(all_clusters)
    n_clusters = len(clusters_sorted)
    cluster_to_idx = {c: i for i, c in enumerate(clusters_sorted)}

    n_clients = len(client_ids)
    matrix = np.zeros((n_clients, n_clusters), dtype=np.int64)
    for row, cid in enumerate(client_ids):
        for c, count in counts_per_client[cid].items():
            matrix[row, cluster_to_idx[c]] = count

    # Global distribution scaled to average client size
    client_totals = matrix.sum(axis=1)
    global_counts = matrix.sum(axis=0)
    global_total = float(global_counts.sum()) or 1.0
    global_frac = global_counts / global_total
    avg_client_total = float(client_totals.mean()) or 1.0
    global_scaled = global_frac * avg_client_total

    num_bars = n_clients + 1  # global + per-client
    fig, ax = plt.subplots(figsize=(max(12, num_bars * 1.5), 7))
    x = np.arange(num_bars)
    bottom = np.zeros(num_bars)

    cmap = plt.cm.tab20 if n_clusters <= 20 else plt.cm.tab20b
    colors = cmap(np.linspace(0, 1, n_clusters))

    for i, c in enumerate(clusters_sorted):
        heights = np.zeros(num_bars)
        heights[0] = global_scaled[i]
        for row, cid in enumerate(client_ids):
            heights[row + 1] = matrix[row, cluster_to_idx[c]]
        ax.bar(x, heights, width=0.75, bottom=bottom, color=colors[i], label=f"Cluster {c}")
        bottom += heights

    # Separate IID and non-IID clients
    client_types = [client_data[cid]["type"] for cid in client_ids]
    num_iid = sum(1 for t in client_types if t == "iid")
    num_noniid = n_clients - num_iid

    # Build x-axis labels with group headers (bigger font)
    x_labels = ["Global"]
    for i, (cid, ctype) in enumerate(zip(client_ids, client_types)):
        x_labels.append(f"C{cid}\n({ctype[:3]})")

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=11, weight="bold")

    # Add shaded background regions for IID vs non-IID groups
    iid_end = 1 + num_iid - 0.5
    noniid_start = 1 + num_iid - 0.5
    ax.axvspan(-0.5, iid_end, alpha=0.08, color="blue", zorder=0)
    if num_noniid > 0:
        ax.axvspan(noniid_start, num_bars - 0.5, alpha=0.08, color="red", zorder=0)

    # Add alpha value labels above each group (bigger font)
    if num_iid > 0:
        iid_center = (0 + iid_end + 0.5) / 2  # Center of IID group
        max_height = bottom[min(int(iid_center), num_bars - 1)]
        ax.text(iid_center, max_height * 1.08, f"IID Group\nα={alpha_iid}",
                ha="center", fontsize=12, weight="bold", color="#0d47a1",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.85, edgecolor="#0d47a1", linewidth=2))

    if num_noniid > 0:
        noniid_center = (noniid_start + num_bars - 0.5) / 2  # Center of non-IID group
        max_height = max(bottom[max(1 + num_iid, 0):]) if 1 + num_iid < len(bottom) else max(bottom)
        ax.text(noniid_center, max_height * 1.08, f"Non-IID Group\nα={alpha_noniid}",
                ha="center", fontsize=12, weight="bold", color="#b71c1c",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcoral", alpha=0.85, edgecolor="#b71c1c", linewidth=2))

    ax.set_ylabel("Number of samples", fontsize=13, weight="bold")
    ax.set_title("Client Data Distribution (Train Set)", fontsize=15, weight="bold", pad=20)

    # Style y-axis
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.5)

    # Bigger legend outside
    if n_clusters <= 20:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9, ncol=1, frameon=True, shadow=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Distribution plot saved: {output_path}")
