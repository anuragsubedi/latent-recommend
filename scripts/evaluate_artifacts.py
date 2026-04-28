"""Compute M2 evaluation metrics and update dashboard-ready artifact metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, normalized_mutual_info_score, silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from latent_recommend.analysis import add_optional_umap, compute_projection_frame
from latent_recommend.config import ArtifactPaths
from latent_recommend.db import connect, initialize_schema, load_tracks, update_projection_columns, upsert_evaluation
from latent_recommend.metrics import (
    centroid_separation,
    mean_reciprocal_rank,
    precision_at_k,
    triplet_success_rate,
)
from latent_recommend.playlists import (
    evaluate_playlist_completion,
    generate_acoustic_playlists,
    playlist_summary_json,
    write_playlists,
)
from latent_recommend.retrieval import RetrievalEngine, normalize_embeddings, save_faiss_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--clusters", type=int, default=None)
    parser.add_argument("--playlist-count", type=int, default=30)
    parser.add_argument("--playlist-k", type=int, default=4)
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
    artifact_integrity = {
        "metadata_rows": int(len(tracks)),
        "embedding_shape": list(embeddings.shape),
        "embeddings_finite": bool(np.isfinite(embeddings).all()),
        "faiss_ids_contiguous": tracks["faiss_id"].tolist() == list(range(len(tracks))),
        "preview_rows": int(tracks["preview_path"].notna().sum()),
    }

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

    try:
        save_faiss_index(embeddings, paths.index_path, metric="cosine")
    except ImportError:
        print("faiss is not installed locally; continuing with NumPy retrieval only.")

    tracks = load_tracks(paths.metadata_path)
    engine = RetrievalEngine(
        tracks=tracks,
        embeddings=embeddings,
        index_path=paths.index_path if paths.index_path.exists() else None,
        metric="cosine",
    )

    precision_at_4 = precision_at_k(engine, k=4, mode="faiss")
    precision = precision_at_k(engine, k=args.k, mode="faiss")
    playlists = generate_acoustic_playlists(
        engine,
        playlist_count=args.playlist_count,
        seed=42,
        mode="raw64",
    )
    playlist_eval = evaluate_playlist_completion(
        engine,
        playlists,
        k=args.playlist_k,
        mode="raw64",
        holdout_strategy="centroid",
    )
    with connect(paths.metadata_path) as conn:
        initialize_schema(conn)
        write_playlists(conn, playlists)

    cluster_labels = tracks["cluster"].fillna(-1).astype(int).to_numpy()
    proxy_codes = tracks["primary_tag"].astype("category").cat.codes.to_numpy()
    clustering = {}
    valid_cluster_count = len(set(cluster_labels))
    if valid_cluster_count > 1:
        clustering["silhouette"] = float(silhouette_score(embeddings, cluster_labels))
        clustering["davies_bouldin"] = float(davies_bouldin_score(embeddings, cluster_labels))
        clustering["adjusted_rand_proxy"] = float(adjusted_rand_score(proxy_codes, cluster_labels))
        clustering["normalized_mutual_info_proxy"] = float(normalized_mutual_info_score(proxy_codes, cluster_labels))

    metrics = {
        "dataset": {
            "tracks": int(len(tracks)),
            "embedding_dimensions": int(embeddings.shape[1]),
            "tags": tracks["primary_tag"].value_counts().to_dict(),
        },
        "artifact_integrity": artifact_integrity,
        "retrieval": {
            "precision_at_4": precision_at_4["macro"],
            f"precision_at_{args.k}": precision["macro"],
            f"mrr_at_{args.k}": mean_reciprocal_rank(engine, k=args.k, mode="faiss"),
            "precision_by_tag": precision["per_tag"],
        },
        "playlist_completion": playlist_eval["summary"],
        "generated_playlists": {
            "count": len(playlists),
            "size_min": min((len(playlist["tracks"]) for playlist in playlists), default=0),
            "size_max": max((len(playlist["tracks"]) for playlist in playlists), default=0),
            "strategy": "centroid_cohesive_acoustic",
            "holdout_strategy": "centroid",
        },
        "ablation": {},
        "topology": {
            "centroid": centroid_separation(embeddings, tracks),
            "triplet_success_rate": triplet_success_rate(embeddings, tracks),
            "clustering": clustering,
        },
        "projection": projection_summary | umap_summary,
    }

    for mode in ("raw64", "pca10", "pca3", "mahalanobis"):
        score = precision_at_k(engine, k=args.k, mode=mode)
        metrics["ablation"][mode] = {f"precision_at_{args.k}": score["macro"]}

    paths.metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True))
    with connect(paths.metadata_path) as conn:
        initialize_schema(conn)
        upsert_evaluation(conn, "metrics_json", None, json.dumps(metrics, sort_keys=True))
        upsert_evaluation(
            conn,
            "generated_playlists",
            float(len(playlists)),
            playlist_summary_json(playlists),
        )
        upsert_evaluation(
            conn,
            "playlist_completion",
            playlist_eval["summary"].get(f"precision@{args.playlist_k}"),
            json.dumps(playlist_eval["summary"], sort_keys=True),
        )
        conn.commit()
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
