"""Artifact loading and small demo fallback data for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from latent_recommend.config import ArtifactPaths
from latent_recommend.db import load_tracks
from latent_recommend.retrieval import normalize_embeddings


def artifacts_available(paths: ArtifactPaths) -> bool:
    return paths.metadata_path.exists() and (
        paths.embeddings_path.exists() or paths.index_path.exists()
    )


def load_embeddings(paths: ArtifactPaths) -> np.ndarray | None:
    if paths.embeddings_path.exists():
        return normalize_embeddings(np.load(paths.embeddings_path))
    return None


def load_metrics(paths: ArtifactPaths) -> dict:
    if not paths.metrics_path.exists():
        return {}
    return json.loads(paths.metrics_path.read_text())


def load_metadata(paths: ArtifactPaths) -> pd.DataFrame:
    if paths.metadata_path.exists():
        return load_tracks(paths.metadata_path)
    return demo_tracks()


def demo_tracks() -> pd.DataFrame:
    rows = [
        ("ambient", "Long Pad Drift", "Fixture Artist A", -1.2, 0.1, 0.3),
        ("ambient", "Low Cloud Loop", "Fixture Artist B", -1.0, 0.2, 0.4),
        ("dub", "Sub Delay Study", "Fixture Artist C", 0.6, -0.4, 0.2),
        ("electronic", "Glitch Grid", "Fixture Artist D", 0.9, 0.5, -0.2),
        ("classical", "String Room", "Fixture Artist E", -0.3, 1.1, -0.5),
        ("heavy_metal", "Bright Distortion", "Fixture Artist F", 1.4, -1.0, 0.6),
    ]
    data = []
    for idx, (tag, title, artist, x, y, z) in enumerate(rows):
        data.append(
            {
                "faiss_id": idx,
                "track_id": f"fixture-{idx}",
                "title": title,
                "artist": artist,
                "album": "Fixture Set",
                "duration": 30.0,
                "primary_tag": tag,
                "tags": tag,
                "audio_url": None,
                "preview_path": None,
                "split": "demo",
                "pca_1": x,
                "pca_2": y,
                "pca_3": z,
                "cluster": idx % 3,
            }
        )
    return pd.DataFrame(data)


def demo_embeddings(track_count: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return normalize_embeddings(rng.normal(size=(track_count, 64)).astype("float32"))


def write_manifest(paths: ArtifactPaths, payload: dict) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
