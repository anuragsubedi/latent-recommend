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
        ("ambient_soundscape", "Long Pad Drift", "Fixture Artist A", "Ambient / atmospheric / meditative", -1.2, 0.1, 0.3),
        ("electronic_dance", "Sub Pulse Study", "Fixture Artist B", "Electronic / techno / dance", -1.0, 0.2, 0.4),
        ("classical_orchestral", "String Room", "Fixture Artist C", "Classical / orchestral", 0.6, -0.4, 0.2),
        ("jazz_soul", "Blue Interval", "Fixture Artist D", "Jazz / soul", 0.9, 0.5, -0.2),
        ("hiphop_rap", "Loop Grid", "Fixture Artist E", "Hiphop / rap / beats", -0.3, 1.1, -0.5),
        ("indie_rock", "Bright Distortion", "Fixture Artist F", "Indie / rock / alternative", 1.4, -1.0, 0.6),
    ]
    data = []
    for idx, (tag, title, artist, tag_blob, x, y, z) in enumerate(rows):
        data.append(
            {
                "faiss_id": idx,
                "track_id": f"fixture-{idx}",
                "title": title,
                "display_title": title,
                "artist": artist,
                "artist_id": f"fixture-artist-{idx}",
                "artist_display_name": artist,
                "album": "Fixture Set",
                "album_id": "fixture-album",
                "album_display_title": "Fixture Set",
                "duration": 30.0,
                "primary_tag": tag,
                "tags": json.dumps([s.strip() for s in tag_blob.split("/")]),
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
