"""Configuration shared by the M2 pipeline, evaluator, and Streamlit app."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGET_TAGS = (
    "ambient_soundscape",
    "electronic_dance",
    "classical_orchestral",
    "jazz_soul",
    "blues_roots",
    "hiphop_rap",
    "folk_acoustic",
    "indie_rock",
    "experimental_trip_hop",
    "happy_pop",
)


@dataclass(frozen=True)
class ArtifactPaths:
    """Filesystem layout for generated M2 artifacts.

    The app is intentionally CPU-only and reads these files after the Colab
    extraction pipeline has produced them.
    """

    root: Path = PROJECT_ROOT / "artifacts"

    @property
    def index_path(self) -> Path:
        return self.root / "vectors.index"

    @property
    def metadata_path(self) -> Path:
        return self.root / "metadata.db"

    @property
    def embeddings_path(self) -> Path:
        return self.root / "embeddings.npy"

    @property
    def metrics_path(self) -> Path:
        return self.root / "metrics.json"

    @property
    def previews_dir(self) -> Path:
        return self.root / "previews"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def require(self) -> None:
        missing = [
            str(path)
            for path in (self.index_path, self.metadata_path)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Missing required artifact files: " + ", ".join(missing)
            )
