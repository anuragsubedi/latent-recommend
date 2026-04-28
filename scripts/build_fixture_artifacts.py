"""Build a tiny local artifact bundle for dashboard smoke testing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from latent_recommend.artifacts import demo_embeddings, demo_tracks, write_manifest
from latent_recommend.config import ArtifactPaths
from latent_recommend.db import initialize_schema, insert_tracks, connect
from latent_recommend.retrieval import save_faiss_index


def main() -> None:
    paths = ArtifactPaths()
    paths.root.mkdir(parents=True, exist_ok=True)
    tracks = demo_tracks()
    embeddings = demo_embeddings(len(tracks)).astype("float32")

    with connect(paths.metadata_path) as conn:
        initialize_schema(conn)
        insert_tracks(conn, tracks.to_dict(orient="records"))

    np.save(paths.embeddings_path, embeddings)
    save_faiss_index(embeddings, paths.index_path, metric="cosine")
    write_manifest(
        paths,
        {
            "kind": "fixture",
            "tracks": len(tracks),
            "embedding_dimensions": embeddings.shape[1],
        },
    )
    print(f"Wrote fixture artifacts to {paths.root}")


if __name__ == "__main__":
    main()
