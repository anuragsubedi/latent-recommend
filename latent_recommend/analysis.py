"""Dimensionality reduction and clustering helpers for latent embeddings."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def compute_projection_frame(
    embeddings: np.ndarray,
    tracks: pd.DataFrame,
    n_clusters: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    vectors = np.asarray(embeddings, dtype="float32")
    ordered = tracks.sort_values("faiss_id").reset_index(drop=True).copy()
    scaled = StandardScaler().fit_transform(vectors)

    pca3 = PCA(n_components=min(3, vectors.shape[1]), random_state=42)
    pca3_values = pca3.fit_transform(scaled)
    for idx in range(3):
        ordered[f"pca_{idx + 1}"] = pca3_values[:, idx] if idx < pca3_values.shape[1] else 0.0

    pca10_components = min(10, vectors.shape[1], len(vectors))
    pca10 = PCA(n_components=pca10_components, random_state=42)
    pca10_values = pca10.fit_transform(scaled)
    for idx in range(10):
        ordered[f"pca10_{idx + 1}"] = (
            pca10_values[:, idx] if idx < pca10_values.shape[1] else 0.0
        )

    cluster_count = n_clusters or max(2, min(8, ordered["primary_tag"].nunique()))
    cluster_count = min(cluster_count, len(ordered))
    if cluster_count > 1:
        ordered["cluster"] = KMeans(
            n_clusters=cluster_count,
            random_state=42,
            n_init=10,
        ).fit_predict(scaled)
    else:
        ordered["cluster"] = 0

    summary = {
        "pca3_explained_variance": pca3.explained_variance_ratio_.tolist(),
        "pca10_explained_variance": pca10.explained_variance_ratio_.tolist(),
        "n_clusters": int(cluster_count),
    }
    return ordered, summary


def add_optional_umap(frame: pd.DataFrame, embeddings: np.ndarray) -> tuple[pd.DataFrame, dict]:
    try:
        import umap
    except ImportError:
        frame = frame.copy()
        frame["umap_1"] = None
        frame["umap_2"] = None
        frame["umap_3"] = None
        return frame, {"umap_available": False}

    reducer = umap.UMAP(n_components=3, random_state=42)
    values = reducer.fit_transform(np.asarray(embeddings, dtype="float32"))
    frame = frame.copy()
    frame["umap_1"] = values[:, 0]
    frame["umap_2"] = values[:, 1]
    frame["umap_3"] = values[:, 2]
    return frame, {"umap_available": True}
