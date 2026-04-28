"""Evaluation metrics for proxy playlist completion and latent topology."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd

from latent_recommend.retrieval import RetrievalEngine


def precision_at_k(engine: RetrievalEngine, k: int = 10, mode: str = "faiss") -> dict:
    per_tag_hits: dict[str, list[float]] = defaultdict(list)
    rows = []

    for _, seed in engine.tracks.iterrows():
        seed_id = int(seed["faiss_id"])
        seed_tag = seed["primary_tag"]
        results = engine.query(seed_id, k=k, mode=mode)
        if results.empty:
            continue
        hits = (results.head(k)["primary_tag"] == seed_tag).astype(float)
        score = float(hits.mean())
        per_tag_hits[str(seed_tag)].append(score)
        rows.append({"faiss_id": seed_id, "primary_tag": seed_tag, f"precision@{k}": score})

    per_tag = {
        tag: float(np.mean(scores))
        for tag, scores in sorted(per_tag_hits.items())
        if scores
    }
    macro = float(np.mean(list(per_tag.values()))) if per_tag else 0.0
    detail = pd.DataFrame(rows)
    return {"macro": macro, "per_tag": per_tag, "detail": detail}


def mean_reciprocal_rank(engine: RetrievalEngine, k: int = 10, mode: str = "faiss") -> float:
    scores = []
    for _, seed in engine.tracks.iterrows():
        seed_id = int(seed["faiss_id"])
        seed_tag = seed["primary_tag"]
        results = engine.query(seed_id, k=k, mode=mode)
        reciprocal = 0.0
        for rank, tag in enumerate(results["primary_tag"].head(k), start=1):
            if tag == seed_tag:
                reciprocal = 1.0 / rank
                break
        scores.append(reciprocal)
    return float(np.mean(scores)) if scores else 0.0


def centroid_separation(embeddings: np.ndarray, tracks: pd.DataFrame) -> dict:
    vectors = np.asarray(embeddings, dtype="float32")
    labels = tracks.sort_values("faiss_id")["primary_tag"].to_numpy()
    centroids = {
        tag: vectors[labels == tag].mean(axis=0)
        for tag in sorted(set(labels))
        if np.any(labels == tag)
    }

    intra = {}
    for tag, centroid in centroids.items():
        group = vectors[labels == tag]
        intra[tag] = float(np.linalg.norm(group - centroid, axis=1).mean())

    inter_values = [
        float(np.linalg.norm(centroids[left] - centroids[right]))
        for left, right in combinations(centroids, 2)
    ]
    inter_mean = float(np.mean(inter_values)) if inter_values else 0.0
    intra_mean = float(np.mean(list(intra.values()))) if intra else 0.0
    ratio = inter_mean / intra_mean if intra_mean else 0.0
    return {
        "intra_mean": intra_mean,
        "inter_mean": inter_mean,
        "separation_ratio": float(ratio),
        "intra_by_tag": intra,
    }


def triplet_success_rate(embeddings: np.ndarray, tracks: pd.DataFrame, samples_per_tag: int = 100) -> float:
    rng = np.random.default_rng(42)
    vectors = np.asarray(embeddings, dtype="float32")
    ordered = tracks.sort_values("faiss_id").reset_index(drop=True)
    groups = {
        tag: ordered.index[ordered["primary_tag"] == tag].to_numpy()
        for tag in sorted(ordered["primary_tag"].dropna().unique())
    }
    successes = 0
    total = 0

    for tag, indices in groups.items():
        negative_pool = np.concatenate([idx for other, idx in groups.items() if other != tag])
        if len(indices) < 2 or len(negative_pool) == 0:
            continue
        for _ in range(min(samples_per_tag, len(indices))):
            anchor, positive = rng.choice(indices, size=2, replace=False)
            negative = int(rng.choice(negative_pool))
            positive_distance = np.linalg.norm(vectors[anchor] - vectors[positive])
            negative_distance = np.linalg.norm(vectors[anchor] - vectors[negative])
            successes += int(positive_distance < negative_distance)
            total += 1

    return float(successes / total) if total else 0.0
