"""Normalize merged artifact metadata into track, artist, and album tables."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from latent_recommend.config import ArtifactPaths
from latent_recommend.db import connect, initialize_schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=PROJECT_ROOT / "artifacts" / "merged")
    return parser.parse_args()


def add_column(conn: sqlite3.Connection, table: str, column: str, declaration: str) -> None:
    existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
    if column not in existing:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")


def looks_procedural(title: str | None) -> bool:
    if not title:
        return True
    return title.strip().lower().startswith("track ")


def main() -> None:
    args = parse_args()
    paths = ArtifactPaths(root=args.artifacts)
    if not paths.metadata_path.exists():
        raise FileNotFoundError(paths.metadata_path)

    with connect(paths.metadata_path) as conn:
        initialize_schema(conn)
        add_column(conn, "tracks", "display_title", "TEXT")
        add_column(conn, "tracks", "artist_id", "TEXT")
        add_column(conn, "tracks", "album_id", "TEXT")

        rows = conn.execute(
            "SELECT faiss_id, track_id, title, artist, album FROM tracks ORDER BY faiss_id"
        ).fetchall()
        for row in rows:
            track_id = str(row["track_id"])
            artist_id = str(row["artist"] or "unknown")
            album_id = str(row["album"] or "unknown")
            display_title = (
                f"Jamendo Track {track_id}"
                if looks_procedural(row["title"])
                else str(row["title"])
            )
            conn.execute(
                """
                UPDATE tracks
                SET artist_id = ?, album_id = ?, display_title = ?
                WHERE faiss_id = ?
                """,
                (artist_id, album_id, display_title, int(row["faiss_id"])),
            )
            conn.execute(
                "INSERT OR IGNORE INTO artists (artist_id, display_name) VALUES (?, ?)",
                (artist_id, f"Jamendo Artist {artist_id}"),
            )
            conn.execute(
                """
                INSERT OR IGNORE INTO albums (album_id, artist_id, display_title)
                VALUES (?, ?, ?)
                """,
                (album_id, artist_id, f"Jamendo Album {album_id}"),
            )
        conn.commit()

    print(f"Normalized metadata model in {paths.metadata_path}")


if __name__ == "__main__":
    main()
