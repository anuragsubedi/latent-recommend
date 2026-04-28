"""Merge per-tag Colab artifact shards into one dashboard-ready bundle."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from latent_recommend.config import ArtifactPaths
from latent_recommend.db import connect, initialize_schema, insert_track, load_tracks
from latent_recommend.retrieval import normalize_embeddings, save_faiss_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+", type=Path, help="Shard artifact directories.")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument("--copy-previews", action="store_true")
    return parser.parse_args()


def rewrite_preview_path(
    preview_path: str | None,
    source_root: Path,
    output_root: Path,
    new_faiss_id: int,
    copy_previews: bool,
) -> str | None:
    if not preview_path:
        return None

    source_preview = Path(preview_path)
    if not source_preview.is_absolute():
        source_preview = source_root / source_preview
    if not source_preview.exists():
        return preview_path

    suffix = source_preview.suffix or ".mp3"
    destination = output_root / "previews" / f"{new_faiss_id:06d}{suffix}"
    if copy_previews:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_preview, destination)
    return str(destination.relative_to(output_root))


def main() -> None:
    args = parse_args()
    output_paths = ArtifactPaths(root=args.output)
    output_paths.root.mkdir(parents=True, exist_ok=True)
    output_paths.previews_dir.mkdir(parents=True, exist_ok=True)
    for stale_file in (
        output_paths.metadata_path,
        output_paths.embeddings_path,
        output_paths.index_path,
        output_paths.manifest_path,
        output_paths.metrics_path,
    ):
        if stale_file.exists():
            stale_file.unlink()

    merged_embeddings = []
    merged_rows = []
    shard_summaries = []
    seen_track_ids = set()
    duplicate_track_ids = []
    next_faiss_id = 0

    for shard_root in args.shards:
        shard_paths = ArtifactPaths(root=shard_root)
        if not shard_paths.metadata_path.exists():
            raise FileNotFoundError(f"Missing shard metadata: {shard_paths.metadata_path}")
        if not shard_paths.embeddings_path.exists():
            raise FileNotFoundError(f"Missing shard embeddings: {shard_paths.embeddings_path}")

        tracks = load_tracks(shard_paths.metadata_path)
        embeddings = np.load(shard_paths.embeddings_path).astype("float32")
        if len(tracks) != embeddings.shape[0]:
            raise ValueError(
                f"{shard_root}: {len(tracks)} metadata rows but {embeddings.shape[0]} embeddings."
            )

        for local_idx, (_, row) in enumerate(tracks.iterrows()):
            track_id = str(row.get("track_id"))
            if track_id in seen_track_ids:
                duplicate_track_ids.append(
                    {
                        "track_id": track_id,
                        "source": str(shard_root),
                        "primary_tag": row.get("primary_tag"),
                    }
                )
                continue
            seen_track_ids.add(track_id)

            new_row = row.to_dict()
            new_row["faiss_id"] = next_faiss_id
            new_row["preview_path"] = rewrite_preview_path(
                new_row.get("preview_path"),
                source_root=shard_paths.root,
                output_root=output_paths.root,
                new_faiss_id=next_faiss_id,
                copy_previews=args.copy_previews,
            )
            merged_rows.append(new_row)
            merged_embeddings.append(embeddings[local_idx])
            next_faiss_id += 1

        manifest = {}
        if shard_paths.manifest_path.exists():
            manifest = json.loads(shard_paths.manifest_path.read_text())
        shard_summaries.append(
            {
                "path": str(shard_root),
                "tracks": int(len(tracks)),
                "tags": tracks["primary_tag"].value_counts().to_dict(),
                "manifest": manifest,
            }
        )

    merged_matrix = normalize_embeddings(np.vstack(merged_embeddings).astype("float32"))
    with connect(output_paths.metadata_path) as conn:
        initialize_schema(conn)
        for row in merged_rows:
            insert_track(conn, row)
        conn.commit()

    np.save(output_paths.embeddings_path, merged_matrix)
    faiss_index_written = True
    try:
        save_faiss_index(merged_matrix, output_paths.index_path, metric="cosine")
    except ImportError:
        faiss_index_written = False
        print("faiss is not installed locally; skipped merged vectors.index creation.")
    output_paths.manifest_path.write_text(
        json.dumps(
            {
                "kind": "merged_shards",
                "tracks": int(len(merged_rows)),
                "embedding_dimensions": int(merged_matrix.shape[1]),
                "faiss_index_written": faiss_index_written,
                "duplicate_tracks_skipped": len(duplicate_track_ids),
                "duplicate_track_examples": duplicate_track_ids[:20],
                "shards": shard_summaries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(
        f"Merged {len(merged_rows)} tracks into {output_paths.root}; "
        f"skipped {len(duplicate_track_ids)} duplicate track IDs."
    )


if __name__ == "__main__":
    main()
