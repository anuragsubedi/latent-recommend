"""Generate the poster/dashboard pipeline architecture diagram."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "artifacts" / ".mplconfig"))

import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


OUT_DIR = PROJECT_ROOT / "documentations" / "figures"
PNG_PATH = OUT_DIR / "latent_recommend_pipeline_architecture.png"
SVG_PATH = OUT_DIR / "latent_recommend_pipeline_architecture.svg"

BG = "#0f1117"
PANEL = "#171b25"
PANEL_2 = "#202636"
TEXT = "#f5f7fb"
MUTED = "#aeb7c6"
BLUE = "#7cc7ff"
RED = "#ff4b5c"
GREEN = "#63d297"
YELLOW = "#ffd166"
PURPLE = "#9b8cff"
ORANGE = "#ffb86b"
TEAL = "#5dd6c7"
GRID = "#2b3344"


def glow(patch, color: str) -> None:
    patch.set_path_effects(
        [
            patheffects.SimplePatchShadow(offset=(0, -1), alpha=0.22, rho=0.95),
            patheffects.Normal(),
        ]
    )


def rounded_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    color: str = BLUE,
    fill: str = PANEL_2,
    fontsize: int = 10,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.13",
        linewidth=1.4,
        edgecolor=color,
        facecolor=fill,
        alpha=0.98,
    )
    ax.add_patch(patch)
    glow(patch, color)
    ax.text(x + 0.16, y + h - 0.28, title, color=TEXT, fontsize=fontsize, weight="bold", va="top")
    if subtitle:
        ax.text(
            x + 0.16,
            y + h - 0.62,
            subtitle,
            color=MUTED,
            fontsize=fontsize - 2,
            va="top",
            linespacing=1.12,
        )
    return patch


def group_box(ax, x: float, y: float, w: float, h: float, label: str, edge: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.04,rounding_size=0.22",
        linewidth=1.1,
        edgecolor=edge,
        facecolor="#121620",
        alpha=0.75,
        linestyle="--",
    )
    ax.add_patch(patch)
    ax.text(x + 0.18, y + h - 0.22, label.upper(), color=edge, fontsize=8.5, weight="bold", va="top")


def arrow(ax, start: tuple[float, float], end: tuple[float, float], color: str = MUTED, rad: float = 0.0) -> None:
    patch = FancyArrowPatch(
        start,
        end,
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.8,
        color=color,
        shrinkA=4,
        shrinkB=4,
    )
    ax.add_patch(patch)


def label(ax, x: float, y: float, text: str, color: str = MUTED, size: int = 8, align: str = "center") -> None:
    ax.text(x, y, text, color=color, fontsize=size, ha=align, va="center", linespacing=1.2)


def waveform(ax, x: float, y: float, w: float, h: float) -> None:
    rect = Rectangle((x, y), w, h, facecolor="#07140f", edgecolor=GREEN, linewidth=1.0)
    ax.add_patch(rect)
    xs = [x + w * i / 70 for i in range(71)]
    for idx, xp in enumerate(xs):
        amp = (0.18 + 0.32 * abs(((idx * 37) % 17) / 16 - 0.5)) * h
        ax.plot([xp, xp], [y + h / 2 - amp, y + h / 2 + amp], color=GREEN, linewidth=0.75, alpha=0.95)


def trapezoid(ax, x: float, y: float, w: float, h: float, title: str, subtitle: str, color: str) -> None:
    poly = Polygon(
        [(x, y + 0.08), (x + w * 0.82, y), (x + w, y + h / 2), (x + w * 0.82, y + h), (x, y + h - 0.08)],
        closed=True,
        facecolor=PANEL_2,
        edgecolor=color,
        linewidth=1.5,
    )
    ax.add_patch(poly)
    glow(poly, color)
    ax.text(x + 0.18, y + h - 0.28, title, color=TEXT, fontsize=10, weight="bold", va="top")
    ax.text(x + 0.18, y + h - 0.65, subtitle, color=MUTED, fontsize=8, va="top", linespacing=1.2)


def draw() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10.2), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10.2)
    ax.axis("off")

    ax.text(
        0.55,
        9.72,
        "latent-recommend pipeline architecture",
        color=TEXT,
        fontsize=25,
        weight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.58,
        9.27,
        "Content-first recommendation: raw audio -> ACE-Step VAE latent geometry -> synthetic playlists -> centroid completion.",
        color=MUTED,
        fontsize=11,
        ha="left",
        va="top",
    )

    group_box(ax, 0.45, 0.85, 3.0, 7.75, "open music inputs", BLUE)
    group_box(ax, 3.8, 0.85, 3.0, 7.75, "sampling contract", GREEN)
    group_box(ax, 7.15, 0.85, 4.05, 7.75, "vae-only extraction", ORANGE)
    group_box(ax, 11.55, 0.85, 2.55, 7.75, "artifact bundle", PURPLE)
    group_box(ax, 14.45, 5.05, 3.05, 3.55, "interactive recommender", TEAL)
    group_box(ax, 14.45, 0.85, 3.05, 3.65, "playlist completion", RED)

    rounded_box(ax, 0.8, 7.2, 2.25, 0.92, "MTG-Jamendo", "Open music audio\nstreamed from HF tar shards", BLUE)
    waveform(ax, 1.05, 6.35, 1.72, 0.55)
    label(ax, 1.91, 6.12, "48 kHz stereo waveform", GREEN, 7.5)
    rounded_box(ax, 0.8, 5.02, 2.25, 0.92, "Metadata tags", "Mood, genre, instrument,\nand style labels", BLUE)
    rounded_box(ax, 0.8, 3.8, 2.25, 0.92, "Run config", "200s max duration\n100-track checkpoints", BLUE)

    rounded_box(ax, 4.1, 7.1, 2.35, 1.08, "N-polar sampler", "10 broad proxy buckets\n200 target tracks each", GREEN)
    rounded_box(ax, 4.1, 5.62, 2.35, 1.08, "Two Colab shards", "5 buckets per session\nH100/T4 compatible", GREEN)
    rounded_box(ax, 4.1, 4.15, 2.35, 1.08, "Merge + dedupe", "2,000 raw rows\n1,734 unique tracks", GREEN)
    rounded_box(ax, 4.1, 2.52, 2.35, 1.08, "Proxy buckets", "Weak playlist contexts\nused for sampling + QA", GREEN)

    rounded_box(ax, 7.45, 7.06, 1.95, 0.92, "Decode audio", "Opus -> waveform\nresample and crop", ORANGE)
    trapezoid(ax, 7.45, 5.35, 2.2, 1.18, "ACE-Step 1.5 VAE", "AutoencoderOobleck\nsubfolder='vae'", ORANGE)
    rounded_box(ax, 9.05, 3.82, 1.82, 0.98, "Pool latent time", "Mean over latent frames\none song vector", ORANGE)
    rounded_box(ax, 7.45, 2.28, 2.05, 0.98, "Normalize", "L2-normalized\n64-D embedding", ORANGE)
    rounded_box(ax, 9.72, 6.28, 1.08, 1.4, "LM/DiT", "not loaded\nfor M2", "#6b7280", "#151820", 9)
    label(ax, 10.26, 5.96, "generation stack bypassed", "#7c8494", 7.5)

    rounded_box(ax, 11.88, 7.0, 1.88, 0.86, "embeddings.npy", "1,734 x 64 float matrix", PURPLE)
    rounded_box(ax, 11.88, 5.82, 1.88, 0.86, "metadata.db", "tracks, artists, albums,\nplaylists, evaluations", PURPLE)
    rounded_box(ax, 11.88, 4.54, 1.88, 0.86, "vectors.index", "FAISS cosine index", PURPLE)
    rounded_box(ax, 11.88, 3.35, 1.88, 0.86, "previews/*.mp3", "15 second snippets", PURPLE)
    rounded_box(ax, 11.88, 2.08, 1.88, 0.86, "metrics.json", "retrieval, topology,\nplaylist completion", PURPLE)

    rounded_box(ax, 14.78, 7.15, 2.38, 0.92, "Seed selection", "track, artist, album,\nor playlist context", TEAL)
    rounded_box(ax, 14.78, 6.0, 2.38, 0.92, "Similarity search", "single seed or\nmulti-seed centroid", TEAL)
    rounded_box(ax, 14.78, 4.92, 2.38, 0.9, "Streamlit UI", "audio previews\nPCA topology", TEAL)

    rounded_box(ax, 14.78, 3.45, 2.38, 0.92, "Generate playlists", "30 playlists, 5-9 songs\nnearest latent neighbors", RED)
    rounded_box(ax, 14.78, 2.25, 2.38, 0.92, "Holdout completion", "hide 1-2 tracks\nquery centroid", RED)
    rounded_box(ax, 14.78, 1.04, 2.38, 1.0, "Evaluate", "Precision@4, Recall@4\nHitRate@4, MRR@4", RED)

    arrow(ax, (3.05, 7.66), (4.08, 7.66), BLUE)
    arrow(ax, (3.05, 5.48), (4.08, 7.34), BLUE, rad=0.16)
    arrow(ax, (3.05, 4.26), (4.08, 6.07), BLUE, rad=0.18)
    arrow(ax, (5.28, 7.1), (5.28, 6.72), GREEN)
    arrow(ax, (5.28, 5.62), (5.28, 5.24), GREEN)
    arrow(ax, (5.28, 4.15), (5.28, 3.62), GREEN)
    arrow(ax, (6.45, 4.7), (7.42, 7.38), GREEN, rad=-0.22)
    arrow(ax, (8.42, 7.06), (8.47, 6.56), ORANGE)
    arrow(ax, (9.62, 5.94), (10.0, 4.82), ORANGE, rad=-0.12)
    arrow(ax, (9.8, 3.82), (8.48, 3.28), ORANGE, rad=-0.1)
    arrow(ax, (9.5, 2.78), (11.86, 7.35), PURPLE, rad=-0.28)
    arrow(ax, (9.5, 2.78), (11.86, 6.18), PURPLE, rad=-0.2)
    arrow(ax, (9.5, 2.78), (11.86, 4.9), PURPLE, rad=-0.1)
    arrow(ax, (13.76, 7.42), (14.76, 7.62), TEAL)
    arrow(ax, (13.76, 4.98), (14.76, 6.45), TEAL, rad=0.14)
    arrow(ax, (13.76, 5.15), (14.76, 3.88), RED, rad=-0.12)
    arrow(ax, (15.96, 7.15), (15.96, 6.95), TEAL)
    arrow(ax, (15.96, 6.0), (15.96, 5.77), TEAL)
    arrow(ax, (15.96, 3.45), (15.96, 3.2), RED)
    arrow(ax, (15.96, 2.25), (15.96, 1.99), RED)

    label(ax, 8.68, 4.98, "latent frames", ORANGE, 7.8)
    label(ax, 10.55, 2.95, "64-D content vector", ORANGE, 7.8)
    label(ax, 14.18, 6.72, "retrieval path", TEAL, 7.8)
    label(ax, 14.2, 3.84, "playlist path", RED, 7.8)

    ax.text(
        0.62,
        0.28,
        "Canonical facts: 10 proxy buckets | 1,734 unique tracks | 64-D ACE-Step VAE embeddings | 30 generated playlists | CPU-only dashboard.",
        color=MUTED,
        fontsize=9.5,
        ha="left",
        va="bottom",
    )

    fig.savefig(PNG_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(SVG_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    draw()
