"""Create poster-ready runtime profiling plots from Colab JSONL logs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("artifacts") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path, help="runtime_profile.jsonl files.")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    return parser.parse_args()


def load_profile(path: Path) -> pd.DataFrame:
    rows = []
    with path.open() as profile_file:
        for line in profile_file:
            if line.strip():
                row = json.loads(line)
                row["profile"] = path.parent.name
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([load_profile(path) for path in args.profiles], ignore_index=True)

    timing_columns = ["decode_seconds", "encode_seconds", "preview_seconds", "db_seconds"]
    summary = frame.groupby("profile")[timing_columns + ["total_seconds"]].mean()
    summary.to_csv(args.output_dir / "runtime_profile_summary.csv")

    ax = summary[timing_columns].plot(kind="bar", stacked=True, figsize=(10, 6))
    ax.set_title("Runtime Profile by Extraction Shard")
    ax.set_xlabel("Shard")
    ax.set_ylabel("Mean seconds per track")
    ax.legend(["Decode", "Encode", "Preview", "SQLite"], loc="upper right")
    plt.tight_layout()
    plt.savefig(args.output_dir / "runtime_profile_stacked.png", dpi=200)

    rolling = frame.sort_values(["profile", "n"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for profile, group in rolling.groupby("profile"):
        ax.plot(group["n"], group["rolling_avg_seconds"], label=profile)
    ax.set_title("Rolling Extraction Throughput")
    ax.set_xlabel("Tracks processed")
    ax.set_ylabel("Rolling seconds per track")
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.output_dir / "runtime_profile_rolling.png", dpi=200)

    print(summary)
    print(f"Wrote plots to {args.output_dir}")


if __name__ == "__main__":
    main()
