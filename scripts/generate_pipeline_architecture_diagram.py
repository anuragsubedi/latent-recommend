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
    fig, ax = plt.subplots(figsize=(7.8, 15.5), dpi=220)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 15.5)
    ax.axis("off")

    ax.text(
        0.45,
        15.08,
        "latent-recommend pipeline architecture",
        color=TEXT,
        fontsize=22,
        weight="bold",
        ha="left",
        va="top",
    )
    ax.text(
        0.48,
        14.72,
        "Content-first flow: raw audio -> ACE-Step VAE latents -> retrieval + playlist completion.",
        color=MUTED,
        fontsize=10.8,
        ha="left",
        va="top",
    )

    left_x = 0.45
    left_w = 3.25
    right_x = 4.15
    right_w = 3.4
    lane_bottom = 1.05
    lane_top = 14.05
    lane_h = lane_top - lane_bottom

    group_box(ax, left_x, lane_bottom, left_w, lane_h, "dataset + extraction", ORANGE)
    group_box(ax, right_x, lane_bottom, right_w, lane_h, "artifacts + product surfaces", TEAL)

    rounded_box(
        ax,
        0.7,
        12.65,
        2.75,
        1.05,
        "MTG-Jamendo inputs",
        "Open music audio + metadata\nstreamed from HF tar shards",
        BLUE,
    )
    waveform(ax, 0.95, 12.0, 2.25, 0.45)
    label(ax, 2.05, 11.78, "48 kHz stereo waveform", GREEN, 7.4)
    rounded_box(
        ax,
        0.7,
        11.0,
        2.75,
        0.95,
        "Sampling contract",
        "10 proxy buckets\n200 target tracks each",
        GREEN,
    )
    rounded_box(
        ax,
        0.7,
        9.8,
        2.75,
        0.95,
        "Merge + dedupe",
        "2,000 raw rows ->\n1,734 unique tracks",
        GREEN,
    )
    rounded_box(
        ax,
        0.7,
        8.45,
        2.75,
        1.05,
        "Decode audio",
        "Opus -> waveform\nresample + crop",
        ORANGE,
    )
    trapezoid(
        ax,
        0.7,
        7.0,
        2.9,
        1.15,
        "ACE-Step 1.5 VAE",
        "AutoencoderOobleck\nsubfolder='vae'",
        ORANGE,
    )
    rounded_box(
        ax,
        0.7,
        5.75,
        2.75,
        1.0,
        "Pool latent time",
        "Mean over latent frames\none song vector",
        ORANGE,
    )
    rounded_box(
        ax,
        0.7,
        4.5,
        2.75,
        1.0,
        "Normalize",
        "L2-normalized\n64-D embedding",
        ORANGE,
    )
    rounded_box(
        ax,
        1.52,
        3.45,
        1.5,
        0.86,
        "LM/DiT",
        "not loaded\nfor M2",
        "#6b7280",
        "#151820",
        9,
    )
    label(ax, 2.28, 3.18, "generation stack bypassed", "#7c8494", 7.3)

    rounded_box(
        ax,
        4.45,
        11.95,
        2.8,
        1.0,
        "Artifact bundle",
        "embeddings.npy + metadata.db\nvectors.index + previews + metrics",
        PURPLE,
    )
    rounded_box(
        ax,
        4.45,
        10.4,
        2.8,
        1.0,
        "Interactive recommender",
        "seed selection ->\nsimilarity search -> Streamlit UI",
        TEAL,
    )
    rounded_box(
        ax,
        4.45,
        9.0,
        2.8,
        1.0,
        "Playlist completion",
        "generate 30 playlists\nnearest latent neighbors",
        RED,
    )
    rounded_box(
        ax,
        4.45,
        7.8,
        2.8,
        0.92,
        "Holdout completion",
        "hide 1-2 tracks\nquery centroid",
        RED,
    )
    rounded_box(
        ax,
        4.45,
        6.7,
        2.8,
        0.92,
        "Evaluation",
        "Precision@4, Recall@4\nHitRate@4, MRR@4",
        RED,
    )

    rounded_box(
        ax,
        4.45,
        4.7,
        2.8,
        1.55,
        "Middle-column fit target",
        "Portrait-first layout for poster center column\n(minimal crossing arrows + larger labels).\nUse this figure as the architecture anchor.",
        "#c6d0e3",
        "#1a2130",
        9,
    )

    arrow(ax, (2.06, 12.65), (2.06, 11.95), GREEN)
    arrow(ax, (2.06, 11.0), (2.06, 10.75), GREEN)
    arrow(ax, (2.06, 9.8), (2.06, 9.5), GREEN)
    arrow(ax, (2.06, 8.45), (2.06, 8.16), ORANGE)
    arrow(ax, (2.06, 7.0), (2.06, 6.75), ORANGE)
    arrow(ax, (2.06, 5.75), (2.06, 5.48), ORANGE)
    arrow(ax, (2.06, 4.5), (2.06, 4.28), ORANGE)

    arrow(ax, (3.52, 5.0), (4.42, 12.4), PURPLE, rad=0.24)
    arrow(ax, (5.86, 10.4), (5.86, 10.05), TEAL)
    arrow(ax, (5.86, 9.0), (5.86, 8.75), RED)
    arrow(ax, (5.86, 7.8), (5.86, 7.55), RED)
    arrow(ax, (5.86, 6.7), (5.86, 6.35), RED)

    label(ax, 3.78, 8.55, "64-D content vectors", ORANGE, 7.8, "left")
    label(ax, 5.9, 9.78, "retrieval path", TEAL, 7.8)
    label(ax, 5.9, 8.4, "playlist path", RED, 7.8)

    ax.text(
        0.45,
        0.36,
        "Canonical facts: 10 proxy buckets | 1,734 unique tracks | 64-D embeddings | 30 generated playlists | CPU-only dashboard.",
        color=MUTED,
        fontsize=9.4,
        ha="left",
        va="bottom",
    )

    fig.savefig(PNG_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(SVG_PATH, facecolor=BG, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"Wrote {PNG_PATH}")
    print(f"Wrote {SVG_PATH}")


if __name__ == "__main__":
    draw()
