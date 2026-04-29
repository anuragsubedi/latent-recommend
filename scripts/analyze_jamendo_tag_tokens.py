#!/usr/bin/env python3
"""Summarize distinct Jamendo mood/instrument tags stored in tracks.tags (JSON or text)."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

from latent_recommend.config import ArtifactPaths


def tokenize_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    s = str(raw).strip()
    if not s:
        return []
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x).strip().lower() for x in data if str(x).strip()]
        return [s.lower()]
    except json.JSONDecodeError:
        return [p.strip().lower() for p in s.replace('"', "").split(",") if p.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifact root (default: env LATENT_RECOMMEND_ARTIFACTS or merged bundle).",
    )
    parser.add_argument("--top", type=int, default=60)
    args = parser.parse_args()
    root = args.artifacts or ArtifactPaths().root
    db_path = root / "metadata.db"
    if not db_path.exists():
        raise FileNotFoundError(db_path)

    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT tags FROM tracks WHERE tags IS NOT NULL AND tags != ''").fetchall()
    conn.close()

    counter: Counter[str] = Counter()
    for (cell,) in rows:
        for tok in tokenize_tags(cell):
            counter[tok] += 1

    print(f"tracks_with_tags={len(rows)} unique_tokens={len(counter)}")
    print("rank,count,token")
    for i, (tok, c) in enumerate(counter.most_common(args.top), 1):
        print(f"{i},{c},{tok}")


if __name__ == "__main__":
    main()
