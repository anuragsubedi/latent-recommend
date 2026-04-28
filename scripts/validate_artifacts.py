"""Validate the generated M2 artifact bundle before dashboard deployment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from latent_recommend.config import ArtifactPaths
from latent_recommend.db import load_tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = ArtifactPaths(root=args.artifacts)
    checks: dict[str, object] = {
        "metadata_db": paths.metadata_path.exists(),
        "vectors_index": paths.index_path.exists(),
        "embeddings_npy": paths.embeddings_path.exists(),
        "metrics_json": paths.metrics_path.exists(),
        "previews_dir": paths.previews_dir.exists(),
    }

    if not checks["metadata_db"]:
        raise FileNotFoundError(f"Missing metadata database: {paths.metadata_path}")

    tracks = load_tracks(paths.metadata_path)
    checks["track_count"] = int(len(tracks))
    checks["faiss_ids_contiguous"] = (
        tracks["faiss_id"].tolist() == list(range(len(tracks)))
    )
    checks["primary_tags"] = tracks["primary_tag"].value_counts().to_dict()

    if checks["embeddings_npy"]:
        embeddings = np.load(paths.embeddings_path)
        checks["embedding_shape"] = list(embeddings.shape)
        checks["embedding_dim_ok"] = embeddings.ndim == 2 and embeddings.shape[1] == 64
        checks["embedding_count_matches_metadata"] = embeddings.shape[0] == len(tracks)
        checks["embeddings_finite"] = bool(np.isfinite(embeddings).all())

    preview_paths = tracks["preview_path"].dropna().astype(str)
    existing_previews = 0
    for preview in preview_paths:
        path = Path(preview)
        if not path.is_absolute():
            path = paths.root / path
        existing_previews += int(path.exists())
    checks["preview_rows"] = int(len(preview_paths))
    checks["preview_files_found"] = int(existing_previews)

    failed = [
        name
        for name, value in checks.items()
        if name.endswith("_ok") or name.endswith("_matches_metadata") or name.endswith("_finite")
        if value is False
    ]
    print(json.dumps(checks, indent=2, sort_keys=True))
    if failed:
        raise SystemExit(f"Artifact validation failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
