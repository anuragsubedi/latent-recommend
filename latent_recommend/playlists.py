"""Synthetic playlist generation and multi-seed completion evaluation."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter

import numpy as np
import pandas as pd

from latent_recommend.retrieval import RetrievalEngine


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def _rank_members_by_centroid(
    engine: RetrievalEngine,
    members: list[int],
    mode: str = "raw64",
) -> list[int]:
    vectors = engine._vectors_for_mode(mode)
    member_ids = list(map(int, members))
    centroid = _normalize_vector(vectors[member_ids].mean(axis=0))
    scores = vectors[member_ids] @ centroid
    order = np.argsort(-scores)
    return [member_ids[int(idx)] for idx in order]


def split_playlist_holdout(
    engine: RetrievalEngine,
    members: list[int],
    holdout_count: int = 2,
    mode: str = "raw64",
    strategy: str = "centroid",
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    member_ids = list(map(int, members))
    if len(member_ids) <= holdout_count:
        return member_ids, []
    if strategy == "tail":
        holdouts = member_ids[-holdout_count:]
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        holdouts = rng.choice(member_ids, size=holdout_count, replace=False).astype(int).tolist()
    elif strategy == "centroid":
        holdouts = _rank_members_by_centroid(engine, member_ids, mode=mode)[:holdout_count]
    else:
        raise ValueError(f"Unsupported holdout strategy: {strategy}")
    holdout_set = set(holdouts)
    query_ids = [faiss_id for faiss_id in member_ids if faiss_id not in holdout_set]
    return query_ids, holdouts


def generate_acoustic_playlists(
    engine: RetrievalEngine,
    playlist_count: int = 30,
    min_size: int = 5,
    max_size: int = 9,
    reuse_fraction: float = 1 / 3,
    candidate_pool: int = 35,
    seed: int = 42,
    mode: str = "raw64",
) -> list[dict]:
    rng = np.random.default_rng(seed)
    max_reuse = max(1, int(np.ceil(playlist_count * reuse_fraction)))
    usage: Counter[int] = Counter()
    available = set(engine.tracks["faiss_id"].astype(int).tolist())
    playlists: list[dict] = []

    for playlist_idx in range(playlist_count):
        candidates = [idx for idx in available if usage[idx] < max_reuse]
        if not candidates:
            break
        anchor = int(rng.choice(candidates))
        size = int(rng.integers(min_size, max_size + 1))
        neighbors = engine.query(anchor, k=max(candidate_pool, size * 4), mode=mode)
        pool = [anchor]
        for _, row in neighbors.iterrows():
            candidate = int(row["faiss_id"])
            if usage[candidate] >= max_reuse or candidate in pool:
                continue
            pool.append(candidate)
            if len(pool) >= candidate_pool:
                break
        if len(pool) < min_size:
            continue

        ranked_pool = _rank_members_by_centroid(engine, pool, mode=mode)
        members = []
        for candidate in ranked_pool:
            if usage[candidate] >= max_reuse or candidate in members:
                continue
            members.append(candidate)
            if len(members) >= size:
                break
        if len(members) < min_size:
            continue
        for member in members:
            usage[member] += 1
        playlists.append(
            {
                "playlist_id": f"synthetic_{playlist_idx:03d}",
                "anchor_faiss_id": members[0],
                "strategy": "centroid_cohesive_acoustic",
                "seed": seed,
                "tracks": members,
            }
        )
    return playlists


def write_playlists(conn: sqlite3.Connection, playlists: list[dict]) -> None:
    conn.execute("DELETE FROM playlist_tracks")
    conn.execute("DELETE FROM generated_playlists")
    for playlist in playlists:
        conn.execute(
            """
            INSERT OR REPLACE INTO generated_playlists
            (playlist_id, anchor_faiss_id, strategy, size, seed)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                playlist["playlist_id"],
                playlist["anchor_faiss_id"],
                playlist["strategy"],
                len(playlist["tracks"]),
                playlist["seed"],
            ),
        )
        for position, faiss_id in enumerate(playlist["tracks"]):
            conn.execute(
                """
                INSERT OR REPLACE INTO playlist_tracks
                (playlist_id, faiss_id, position, role)
                VALUES (?, ?, ?, ?)
                """,
                (playlist["playlist_id"], int(faiss_id), position, "member"),
            )
    conn.commit()


def evaluate_playlist_completion(
    engine: RetrievalEngine,
    playlists: list[dict],
    holdout_count: int = 2,
    k: int = 4,
    mode: str = "raw64",
    holdout_strategy: str = "centroid",
) -> dict:
    rows = []
    for playlist in playlists:
        members = list(map(int, playlist["tracks"]))
        if len(members) <= holdout_count:
            continue
        query_ids, holdout_list = split_playlist_holdout(
            engine,
            members,
            holdout_count=holdout_count,
            mode=mode,
            strategy=holdout_strategy,
        )
        holdouts = set(holdout_list)
        results = engine.query_many(query_ids, k=k, mode=mode)
        retrieved = set(results["faiss_id"].astype(int).tolist()) if not results.empty else set()
        hits = len(retrieved & holdouts)
        reciprocal = 0.0
        for rank, faiss_id in enumerate(results["faiss_id"].astype(int).tolist(), start=1):
            if faiss_id in holdouts:
                reciprocal = 1.0 / rank
                break
        rows.append(
            {
                "playlist_id": playlist["playlist_id"],
                "holdout_strategy": holdout_strategy,
                "query_size": len(query_ids),
                "holdout_size": len(holdouts),
                "hits": hits,
                f"precision@{k}": hits / k,
                f"recall@{k}": hits / len(holdouts),
                f"hit_rate@{k}": float(hits > 0),
                f"mrr@{k}": reciprocal,
            }
        )
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"detail": detail, "summary": {}}
    metric_columns = [column for column in detail.columns if "@" in column]
    summary = {column: float(detail[column].mean()) for column in metric_columns}
    summary["playlists"] = int(len(detail))
    return {"detail": detail, "summary": summary}


def playlist_summary_json(playlists: list[dict]) -> str:
    return json.dumps(
        {
            "playlists": len(playlists),
            "sizes": [len(playlist["tracks"]) for playlist in playlists],
        },
        sort_keys=True,
    )
