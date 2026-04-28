"""Nearest-neighbor retrieval over ACE-Step VAE embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import faiss
except ImportError:  # pragma: no cover - allows metric-only environments.
    faiss = None


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    matrix = np.asarray(embeddings, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def build_faiss_index(embeddings: np.ndarray, metric: str = "cosine"):
    if faiss is None:
        raise ImportError("faiss is required to build a vector index.")

    vectors = normalize_embeddings(embeddings) if metric == "cosine" else embeddings
    vectors = np.asarray(vectors, dtype="float32")

    if metric == "cosine":
        index = faiss.IndexFlatIP(vectors.shape[1])
    elif metric == "l2":
        index = faiss.IndexFlatL2(vectors.shape[1])
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    index.add(vectors)
    return index


def save_faiss_index(embeddings: np.ndarray, index_path: Path | str, metric: str = "cosine") -> None:
    if faiss is None:
        raise ImportError("faiss is required to save a vector index.")
    index = build_faiss_index(embeddings, metric=metric)
    faiss.write_index(index, str(index_path))


@dataclass
class RetrievalEngine:
    tracks: pd.DataFrame
    embeddings: np.ndarray | None = None
    index_path: Path | str | None = None
    metric: str = "cosine"

    def __post_init__(self) -> None:
        self.tracks = self.tracks.sort_values("faiss_id").reset_index(drop=True)
        if self.embeddings is not None:
            vectors = np.asarray(self.embeddings, dtype="float32")
            self.embeddings = normalize_embeddings(vectors) if self.metric == "cosine" else vectors
            covariance = np.cov(self.embeddings.T)
            covariance += np.eye(covariance.shape[0]) * 1e-3
            self._mahalanobis_inv = np.linalg.pinv(covariance).astype("float32")
        else:
            self._mahalanobis_inv = None
        self._index = None
        if self.index_path is not None and faiss is not None and Path(self.index_path).exists():
            self._index = faiss.read_index(str(self.index_path))

    def query(self, faiss_id: int, k: int = 10, mode: str = "faiss") -> pd.DataFrame:
        if mode == "faiss" and self._index is not None:
            return self._query_faiss(faiss_id, k)
        return self._query_numpy(faiss_id, k, mode)

    def query_many(self, faiss_ids: list[int], k: int = 10, mode: str = "raw64") -> pd.DataFrame:
        if self.embeddings is None:
            raise ValueError("embeddings are required for multi-seed retrieval.")
        seed_ids = [int(seed_id) for seed_id in faiss_ids]
        if not seed_ids:
            return pd.DataFrame()
        vectors = self._vectors_for_mode(mode)
        query_vector = vectors[seed_ids].mean(axis=0)
        if mode in {"raw64", "faiss"} and self.metric == "cosine":
            norm = np.linalg.norm(query_vector)
            if norm:
                query_vector = query_vector / norm
        return self._query_vector(query_vector, k=k, mode=mode, exclude_ids=set(seed_ids))

    def _query_faiss(self, faiss_id: int, k: int) -> pd.DataFrame:
        if self.embeddings is None:
            query_vector = np.asarray(
                [self._index.reconstruct(int(faiss_id))],
                dtype="float32",
            )
        else:
            query_vector = self.embeddings[int(faiss_id) : int(faiss_id) + 1].astype("float32")
        distances, indices = self._index.search(query_vector, k + 1)
        return self._format_results(indices[0], distances[0], faiss_id)

    def _query_numpy(self, faiss_id: int, k: int, mode: str) -> pd.DataFrame:
        if self.embeddings is None:
            raise ValueError("embeddings are required for NumPy retrieval.")

        vectors = self._vectors_for_mode(mode)
        query_vector = vectors[int(faiss_id)]
        if mode == "mahalanobis":
            delta = vectors - query_vector
            scores = np.sqrt(np.einsum("ij,jk,ik->i", delta, self._mahalanobis_inv, delta))
            order = np.argsort(scores)
            distances = scores[order]
        elif self.metric == "cosine" and mode not in {"pca3", "pca10"}:
            scores = vectors @ query_vector
            order = np.argsort(-scores)
            distances = scores[order]
        else:
            scores = np.linalg.norm(vectors - query_vector, axis=1)
            order = np.argsort(scores)
            distances = scores[order]
        return self._format_results(order[: k + 1], distances[: k + 1], faiss_id)

    def _vectors_for_mode(self, mode: str) -> np.ndarray:
        if self.embeddings is None:
            raise ValueError("embeddings are required for NumPy retrieval.")
        if mode == "pca3":
            columns = ["pca_1", "pca_2", "pca_3"]
            if all(column in self.tracks.columns for column in columns) and not self.tracks[columns].isna().any().any():
                return self.tracks[columns].to_numpy(dtype="float32")
        if mode == "pca10":
            columns = [f"pca10_{idx}" for idx in range(1, 11)]
            if all(column in self.tracks.columns for column in columns) and not self.tracks[columns].isna().any().any():
                return self.tracks[columns].to_numpy(dtype="float32")
        return self.embeddings

    def _query_vector(
        self,
        query_vector: np.ndarray,
        k: int,
        mode: str,
        exclude_ids: set[int] | None = None,
    ) -> pd.DataFrame:
        vectors = self._vectors_for_mode(mode)
        exclude_ids = exclude_ids or set()
        if mode == "mahalanobis":
            delta = vectors - query_vector
            scores = np.sqrt(np.einsum("ij,jk,ik->i", delta, self._mahalanobis_inv, delta))
            order = np.argsort(scores)
            distances = scores[order]
        elif self.metric == "cosine" and mode not in {"pca3", "pca10"}:
            scores = vectors @ query_vector
            order = np.argsort(-scores)
            distances = scores[order]
        else:
            scores = np.linalg.norm(vectors - query_vector, axis=1)
            order = np.argsort(scores)
            distances = scores[order]
        rows = []
        for idx, distance in zip(order, distances):
            idx = int(idx)
            if idx in exclude_ids:
                continue
            row = self.tracks.iloc[idx].to_dict()
            row["rank"] = len(rows) + 1
            row["distance"] = float(distance)
            rows.append(row)
            if len(rows) >= k:
                break
        return pd.DataFrame(rows)

    def _format_results(
        self,
        indices: np.ndarray,
        distances: np.ndarray,
        seed_id: int,
    ) -> pd.DataFrame:
        rows = []
        for rank, (idx, distance) in enumerate(zip(indices, distances), start=0):
            idx = int(idx)
            if idx < 0 or idx == int(seed_id):
                continue
            row = self.tracks.iloc[idx].to_dict()
            row["rank"] = len(rows) + 1
            row["distance"] = float(distance)
            rows.append(row)
        return pd.DataFrame(rows)
