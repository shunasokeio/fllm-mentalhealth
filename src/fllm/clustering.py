"""
Load embeddings from embeddings.npy and cluster them using HDBSCAN or K-means.
Optional dimensionality reduction (PCA or UMAP) before clustering.
"""
import argparse
import os

import numpy as np
import pandas as pd
from datasets import load_dataset, concatenate_datasets
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

try:
    import hdbscan
except ImportError:
    hdbscan = None

try:
    import umap
except ImportError:
    umap = None

DEFAULT_EMBEDDINGS_PATH = "embedded_data/embeddings.npy"
DEFAULT_OUTPUT_DIR = "embedded_data"


def main():
    parser = argparse.ArgumentParser(description="Cluster embeddings with HDBSCAN or K-means")
    parser.add_argument(
        "--method",
        default="hdbscan",
        choices=("hdbscan", "kmeans"),
        help="Clustering method: kmeans (fixed K clusters) or hdbscan (default kmeans)",
    )
    parser.add_argument(
        "--embeddings",
        default=DEFAULT_EMBEDDINGS_PATH,
        help="Path to embeddings.npy",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save cluster_labels.npy",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=20,
        help="Number of clusters for K-means (default 20)",
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=100,
        help="HDBSCAN min_cluster_size (default 5)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="HDBSCAN min_samples (default same as min_cluster_size)",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=("euclidean", "cosine"),
        help="Distance metric for HDBSCAN (use cosine for normalized embeddings)",
    )
    parser.add_argument(
        "--cluster-selection-method",
        default="eom",
        choices=("eom", "leaf"),
        help="HDBSCAN: eom = fewer larger clusters; leaf = more smaller (default leaf)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for K-means (default 42)",
    )
    parser.add_argument(
        "--sweep-k",
        nargs="+",
        type=int,
        default=None,
        help="If set (e.g. 25 50 75 100), run K-means for each K and print silhouette and size std instead of saving labels",
    )
    parser.add_argument(
        "--reduce-dim",
        default="umap",
        choices=("none", "pca", "umap"),
        help="Dimensionality reduction before clustering: none, pca, or umap (default none)",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=50,
        help="Number of dimensions after PCA/UMAP (default 50)",
    )
    parser.add_argument(
        "--dataset",
        default="ShenLab/MentalChat16K",
        help="Dataset name to load for input/output columns (default ShenLab/MentalChat16K)",
    )
    args = parser.parse_args()

    print(f"Loading embeddings from {args.embeddings}...")
    X = np.load(args.embeddings)
    print(f"Shape: {X.shape}")

    # L2-normalize so euclidean ~ cosine; K-means uses euclidean
    X_norm = normalize(X, norm="l2", axis=1)

    # Optional dimensionality reduction before clustering
    if args.reduce_dim == "none":
        X_cluster = X_norm
        print("No dimensionality reduction.")
    elif args.reduce_dim == "pca":
        n_comp = min(args.n_components, X_norm.shape[0], X_norm.shape[1])
        print(f"Reducing dimension with PCA to {n_comp} components...")
        X_cluster = PCA(n_components=n_comp, random_state=args.random_state).fit_transform(X_norm)
        print(f"Reduced shape: {X_cluster.shape}")
    else:  # umap
        if umap is None:
            raise ImportError("Install UMAP with: pip install umap-learn")
        print(f"Reducing dimension with UMAP to {args.n_components} components...")
        reducer = umap.UMAP(
            n_components=min(args.n_components, X_norm.shape[0] - 1),
            random_state=args.random_state,
            metric="cosine",
        )
        X_cluster = reducer.fit_transform(X_norm)
        print(f"Reduced shape: {X_cluster.shape}")

    if args.method == "kmeans":
        if args.sweep_k:
            print(f"Sweeping K values for K-means: {args.sweep_k}")
            for k in args.sweep_k:
                print(f"Running K-means (n_clusters={k}, random_state={args.random_state})...")
                clusterer = KMeans(
                    n_clusters=k,
                    random_state=args.random_state,
                    n_init=10,
                )
                labels = clusterer.fit_predict(X_cluster)
                sil = silhouette_score(X_cluster, labels, metric="euclidean")
                sizes = np.bincount(labels)
                size_std = sizes.std()
                print(f"K={k}: silhouette={sil:.4f}, size_std={size_std:.2f}")
            # In sweep mode we don't save labels; exit after reporting metrics
            return
        else:
            print(f"Running K-means (n_clusters={args.n_clusters}, random_state={args.random_state})...")
            clusterer = KMeans(
                n_clusters=args.n_clusters,
                random_state=args.random_state,
                n_init=10,
            )
            labels = clusterer.fit_predict(X_cluster)
            n_clusters = args.n_clusters
            n_noise = 0
            print(f"Found {n_clusters} clusters (no noise; every point assigned).")
    else:
        if hdbscan is None:
            raise ImportError("Install HDBSCAN with: pip install hdbscan")
        metric = "euclidean" if args.metric == "cosine" else args.metric
        min_samples = args.min_samples if args.min_samples is not None else args.min_cluster_size
        print(f"Running HDBSCAN (min_cluster_size={args.min_cluster_size}, min_samples={min_samples}, metric={metric}, cluster_selection_method={args.cluster_selection_method})...")
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=args.min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_method=args.cluster_selection_method,
        )
        labels = clusterer.fit_predict(X_cluster)
        n_clusters = int(labels.max()) + 1 if labels.max() >= 0 else 0
        n_noise = int((labels == -1).sum())
        print(f"Found {n_clusters} clusters, {n_noise} noise points.")

    # Print cluster sizes
    print("\nCluster sizes:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        if label == -1:
            print(f"  Noise (cluster_id=-1): {count} examples")
        else:
            print(f"  Cluster {label}: {count} examples")
    print(f"Total examples: {len(labels)}")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "cluster_labels.npy")
    np.save(out_path, labels)
    print(f"Saved labels to {out_path}")

    # Load original dataset to get input and output columns
    print(f"Loading dataset {args.dataset} to extract input/output columns...")
    full_dataset = load_dataset(args.dataset)
    dataset = concatenate_datasets([full_dataset[s] for s in full_dataset])
    if len(dataset) != len(labels):
        raise ValueError(
            f"Dataset length {len(dataset)} does not match labels length {len(labels)}"
        )

    # Create dataset with cluster_id, input, output
    print("Creating dataset with cluster_id, input, output columns...")
    df_data = []
    for i in range(len(dataset)):
        row = dataset[i]
        df_data.append({
            "cluster_id": int(labels[i]),
            "input": row.get("input", ""),
            "output": row.get("output", ""),
        })
    
    df = pd.DataFrame(df_data)
    
    # Save as CSV
    csv_path = os.path.join(args.output_dir, "clustered_dataset.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved clustered dataset (cluster_id, input, output) to {csv_path}")
    
    # Also save as HuggingFace dataset
    from datasets import Dataset
    hf_dataset = Dataset.from_pandas(df)
    hf_dataset_path = os.path.join(args.output_dir, "clustered_dataset")
    hf_dataset.save_to_disk(hf_dataset_path)
    print(f"Saved clustered dataset to {hf_dataset_path}")


if __name__ == "__main__":
    main()
