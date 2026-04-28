"""Compute M2 evaluation metrics and update dashboard-ready artifact metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from latent_recommend.analysis import add_optional_umap, compute_projection_frame
from latent_recommend.config import ArtifactPaths
from latent_recommend.db import load_tracks, update_projection_columns
from latent_recommend.metrics import (
    centroid_separation,
    mean_reciprocal_rank,
    precision_at_k,
    triplet_success_rate,
)
from latent_recommend.retrieval import RetrievalEngine, normalize_embeddings, save_faiss_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--skip-umap", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ArtifactPaths(root=args.artifacts)
    if not paths.metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata database: {paths.metadata_path}")
    if not paths.embeddings_path.exists():
        raise FileNotFoundError(f"Missing embeddings file: {paths.embeddings_path}")

    tracks = load_tracks(paths.metadata_path)
    embeddings = normalize_embeddings(np.load(paths.embeddings_path).astype("float32"))

    if len(tracks) != embeddings.shape[0]:
        raise ValueError(
            f"Track count ({len(tracks)}) does not match embeddings ({embeddings.shape[0]})."
        )
    if embeddings.shape[1] != 64:
        raise ValueError(f"Expected 64-D embeddings, received {embeddings.shape[1]} dimensions.")

    projection_frame, projection_summary = compute_projection_frame(
        embeddings,
        tracks,
        n_clusters=args.clusters,
    )
    if not args.skip_umap:
        projection_frame, umap_summary = add_optional_umap(projection_frame, embeddings)
    else:
        umap_summary = {"umap_available": False}

    projection_columns = [
        "pca_1",
        "pca_2",
        "pca_3",
        "cluster",
        "umap_1",
        "umap_2",
        "umap_3",
    ] + [f"pca10_{idx}" for idx in range(1, 11)]
    update_projection_columns(paths.metadata_path, projection_frame, projection_columns)

    save_faiss_index(embeddings, paths.index_path, metric="cosine")

    tracks = load_tracks(paths.metadata_path)
    engine = RetrievalEngine(
        tracks=tracks,
        embeddings=embeddings,
        index_path=paths.index_path,
        metric="cosine",
    )

    precision = precision_at_k(engine, k=args.k, mode="faiss")
    metrics = {
        "dataset": {
            "tracks": int(len(tracks)),
            "embedding_dimensions": int(embeddings.shape[1]),
            "tags": tracks["primary_tag"].value_counts().to_dict(),
        },
        "retrieval": {
            f"precision_at_{args.k}": precision["macro"],
            f"mrr_at_{args.k}": mean_reciprocal_rank(engine, k=args.k, mode="faiss"),
            "precision_by_tag": precision["per_tag"],
        },
        "ablation": {},
        "topology": {
            "centroid": centroid_separation(embeddings, tracks),
            "triplet_success_rate": triplet_success_rate(embeddings, tracks),
        },
        "projection": projection_summary | umap_summary,
    }

    for mode in ("raw64", "pca10", "pca3"):
        score = precision_at_k(engine, k=args.k, mode=mode)
        metrics["ablation"][mode] = {f"precision_at_{args.k}": score["macro"]}

    paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
