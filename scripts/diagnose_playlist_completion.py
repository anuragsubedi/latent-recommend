"""Compare playlist completion protocols for the current artifact bundle."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from latent_recommend.config import ArtifactPaths
from latent_recommend.db import load_tracks
from latent_recommend.playlists import evaluate_playlist_completion, generate_acoustic_playlists
from latent_recommend.retrieval import RetrievalEngine, normalize_embeddings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "merged")
    parser.add_argument("--playlist-count", type=int, default=30)
    parser.add_argument("--k", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ArtifactPaths(root=args.artifacts)
    tracks = load_tracks(paths.metadata_path)
    embeddings = normalize_embeddings(np.load(paths.embeddings_path).astype("float32"))
    engine = RetrievalEngine(
        tracks=tracks,
        embeddings=embeddings,
        index_path=paths.index_path if paths.index_path.exists() else None,
    )
    playlists = generate_acoustic_playlists(
        engine,
        playlist_count=args.playlist_count,
        seed=42,
        mode="raw64",
    )

    print("playlist_count,size_min,size_max,holdout_strategy,precision,recall,hit_rate,mrr")
    sizes = [len(playlist["tracks"]) for playlist in playlists]
    for strategy in ("tail", "random", "centroid"):
        result = evaluate_playlist_completion(
            engine,
            playlists,
            k=args.k,
            mode="raw64",
            holdout_strategy=strategy,
        )
        summary = result["summary"]
        print(
            ",".join(
                [
                    str(len(playlists)),
                    str(min(sizes, default=0)),
                    str(max(sizes, default=0)),
                    strategy,
                    f"{summary.get(f'precision@{args.k}', 0):.4f}",
                    f"{summary.get(f'recall@{args.k}', 0):.4f}",
                    f"{summary.get(f'hit_rate@{args.k}', 0):.4f}",
                    f"{summary.get(f'mrr@{args.k}', 0):.4f}",
                ]
            )
        )


if __name__ == "__main__":
    main()
