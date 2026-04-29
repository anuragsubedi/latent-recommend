"""Narrative Streamlit dashboard for latent-recommend Milestone 2."""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans

from latent_recommend.artifacts import (
    artifacts_available,
    demo_embeddings,
    load_embeddings,
    load_metadata,
    load_metrics,
)
from latent_recommend.config import ArtifactPaths, PROJECT_ROOT
from latent_recommend.playlists import split_playlist_holdout
from latent_recommend.retrieval import RetrievalEngine


st.set_page_config(page_title="latent-recommend", page_icon="music", layout="wide")

PIPELINE_DIAGRAM_PATH = PROJECT_ROOT / "documentations" / "figures" / "latent_recommend_pipeline_architecture.png"


RETRIEVAL_MODES = {
    "faiss": {
        "label": "Raw 64D FAISS",
        "use": "Fast single-seed search over normalized ACE-Step VAE vectors.",
        "distance": "Cosine similarity through FAISS inner product search.",
    },
    "raw64": {
        "label": "Raw 64D cosine",
        "use": "Best default for single-seed and multi-seed centroid recommendation.",
        "distance": "Dot product between L2-normalized 64-D vectors.",
    },
    "pca10": {
        "label": "PCA10",
        "use": "Ablation that keeps the first ten principal components.",
        "distance": "Euclidean distance in the reduced PCA space.",
    },
    "pca3": {
        "label": "PCA3",
        "use": "Visual/debug ablation aligned with the 3D topology plot.",
        "distance": "Euclidean distance in the plotted PCA space.",
    },
    "mahalanobis": {
        "label": "Mahalanobis",
        "use": "Regularized covariance-aware ablation.",
        "distance": "Distance scaled by inverse covariance of the latent corpus.",
    },
}


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --lr-bg: #0f1117;
            --lr-panel: #171a22;
            --lr-panel-2: #1f2430;
            --lr-text: #f4f6fb;
            --lr-muted: #aab2c0;
            --lr-accent: #ff4b5c;
            --lr-accent-2: #7cc7ff;
            --lr-border: rgba(255, 255, 255, 0.10);
        }
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1420px;
        }
        h1 {
            font-size: 3.2rem !important;
            line-height: 1.0 !important;
            letter-spacing: -0.06em;
            margin-bottom: 0.4rem !important;
        }
        h2, h3 {
            letter-spacing: -0.035em;
        }
        .hero {
            border: 1px solid var(--lr-border);
            border-radius: 24px;
            padding: 28px 30px;
            background:
                radial-gradient(circle at top left, rgba(255,75,92,0.18), transparent 32%),
                linear-gradient(135deg, #171a22 0%, #10131a 100%);
            margin-bottom: 1rem;
        }
        .eyebrow {
            color: var(--lr-accent-2);
            text-transform: uppercase;
            letter-spacing: 0.12em;
            font-size: 0.78rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        .hero-title {
            color: var(--lr-text);
            font-size: 2.1rem;
            line-height: 1.12;
            font-weight: 800;
            letter-spacing: -0.045em;
            margin-bottom: 0.6rem;
        }
        .hero-copy, .card-copy {
            color: var(--lr-muted);
            font-size: 1.02rem;
            line-height: 1.55;
        }
        .metric-card, .info-card {
            border: 1px solid var(--lr-border);
            border-radius: 18px;
            padding: 18px 20px;
            background: linear-gradient(180deg, #191d27 0%, #131720 100%);
            min-height: 132px;
        }
        .metric-label {
            color: var(--lr-muted);
            font-size: 0.82rem;
            margin-bottom: 0.35rem;
        }
        .metric-value {
            color: var(--lr-text);
            font-size: 2.05rem;
            font-weight: 800;
            letter-spacing: -0.05em;
        }
        .metric-note {
            color: var(--lr-muted);
            font-size: 0.82rem;
            margin-top: 0.25rem;
        }
        .info-title {
            color: var(--lr-text);
            font-size: 1.05rem;
            font-weight: 750;
            margin-bottom: 0.45rem;
        }
        .section-intro {
            color: var(--lr-muted);
            font-size: 1.0rem;
            line-height: 1.55;
            max-width: 1040px;
            margin-bottom: 1rem;
        }
        .pill-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.4rem 0 1rem 0;
        }
        .pill {
            border: 1px solid var(--lr-border);
            border-radius: 999px;
            padding: 0.32rem 0.65rem;
            background: #202634;
            color: var(--lr-muted);
            font-size: 0.8rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid var(--lr-border);
            border-radius: 16px;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_bundle() -> tuple[ArtifactPaths, RetrievalEngine, pd.DataFrame, dict, bool]:
    paths = ArtifactPaths()
    has_artifacts = artifacts_available(paths)
    tracks = load_metadata(paths)
    embeddings = load_embeddings(paths)
    if embeddings is None:
        embeddings = demo_embeddings(len(tracks))
    engine = RetrievalEngine(
        tracks=tracks,
        embeddings=embeddings,
        index_path=paths.index_path if paths.index_path.exists() else None,
        metric="cosine",
    )
    return paths, engine, tracks, load_metrics(paths), has_artifacts


def load_playlist_tables(metadata_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not metadata_path.exists():
        return pd.DataFrame(), pd.DataFrame()
    with sqlite3.connect(metadata_path) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        if not {"generated_playlists", "playlist_tracks", "tracks"}.issubset(tables):
            return pd.DataFrame(), pd.DataFrame()
        playlists = pd.read_sql_query(
            "SELECT * FROM generated_playlists ORDER BY playlist_id",
            conn,
        )
        playlist_tracks = pd.read_sql_query(
            """
            SELECT
                pt.playlist_id,
                pt.position,
                pt.role,
                t.faiss_id,
                t.track_id,
                COALESCE(t.display_title, t.title, 'Jamendo Track ' || t.track_id) AS track_display_title,
                COALESCE(a.display_name, t.artist, 'Unknown Artist') AS artist_display_name,
                COALESCE(al.display_title, t.album, 'Unknown Album') AS album_display_title,
                t.primary_tag,
                t.tags,
                t.duration,
                t.preview_path,
                t.pca_1,
                t.pca_2,
                t.pca_3,
                t.cluster
            FROM playlist_tracks pt
            JOIN tracks t ON t.faiss_id = pt.faiss_id
            LEFT JOIN artists a ON a.artist_id = COALESCE(t.artist_id, t.artist)
            LEFT JOIN albums al ON al.album_id = COALESCE(t.album_id, t.album)
            ORDER BY pt.playlist_id, pt.position
            """,
            conn,
        )
    return playlists, playlist_tracks


def display_title(row: pd.Series) -> str:
    return str(row.get("track_display_title") or row.get("display_title") or row.get("title") or row.get("track_id"))


def display_artist(row: pd.Series) -> str:
    return str(row.get("artist_display_name") or row.get("artist") or "Unknown Artist")


def display_album(row: pd.Series) -> str:
    return str(row.get("album_display_title") or row.get("album") or "Unknown Album")


def parse_tags_cell(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return [str(x).strip() for x in data if str(x).strip()]
        return [s]
    except json.JSONDecodeError:
        return [p.strip() for p in s.replace('"', "").split(",") if p.strip()]


def format_tags_display(value: object, max_chars: int = 220) -> str:
    tokens = parse_tags_cell(value)
    if not tokens:
        return ""
    text = " · ".join(tokens)
    if len(text) > max_chars:
        return text[: max_chars - 1] + "…"
    return text


def ensure_tags_display(df: pd.DataFrame) -> pd.DataFrame:
    out = df
    if "tags_display" not in out.columns and "tags" in out.columns:
        out = out.copy()
        out["tags_display"] = out["tags"].map(lambda v: format_tags_display(v))
    elif "tags_display" not in out.columns:
        out = out.copy()
        out["tags_display"] = ""
    return out


def jamendo_subtag_vocabulary(tracks: pd.DataFrame, max_tokens: int = 80) -> list[str]:
    if "tags" not in tracks.columns:
        return []
    counts: Counter[str] = Counter()
    for cell in tracks["tags"].tolist():
        for tok in parse_tags_cell(cell):
            counts[str(tok).lower()] += 1
    return [token for token, _ in counts.most_common(max_tokens)]


def apply_centroid_distance_trim(frame: pd.DataFrame, trim_pct: float) -> pd.DataFrame:
    if trim_pct <= 0 or frame.empty or len(frame) < 10:
        return frame
    xyz = frame[["pca_1", "pca_2", "pca_3"]].to_numpy(dtype=np.float64)
    centroid = xyz.mean(axis=0)
    dist = np.linalg.norm(xyz - centroid, axis=1)
    thresh = np.percentile(dist, 100 - trim_pct)
    return frame.iloc[dist <= thresh].copy()


def apply_axis_quantile_trim(frame: pd.DataFrame, q_low: float, q_high: float) -> pd.DataFrame:
    if frame.empty or q_low >= q_high:
        return frame
    out = frame.copy()
    for col in ("pca_1", "pca_2", "pca_3"):
        lo, hi = out[col].quantile([q_low, q_high])
        out = out[(out[col] >= lo) & (out[col] <= hi)]
    return out


def jamendo_tags_haystack(row: pd.Series) -> str:
    return " ".join(parse_tags_cell(row.get("tags"))).lower()


def refine_topology_frame(
    frame: pd.DataFrame,
    proxy_buckets: list[str] | None,
    subtag_tokens: list[str],
    substring: str,
) -> pd.DataFrame:
    out = frame
    if proxy_buckets:
        out = out[out["primary_tag"].isin(proxy_buckets)]
    tag_filter_on = bool(subtag_tokens) or bool(substring.strip())
    if tag_filter_on and "tags" in out.columns:
        masks = []
        if subtag_tokens:
            selected = {t.lower() for t in subtag_tokens}

            def has_token(row: pd.Series) -> bool:
                return any(t.lower() in selected for t in parse_tags_cell(row.get("tags")))

            masks.append(out.apply(has_token, axis=1))
        if substring.strip():
            sub = substring.strip().lower()

            def has_sub(row: pd.Series) -> bool:
                return sub in jamendo_tags_haystack(row)

            masks.append(out.apply(has_sub, axis=1))
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
        out = out[combined]
    return out.copy()


def subgenre_color_labels(frame: pd.DataFrame, selected_tokens: list[str]) -> pd.Series:
    sel = [t.lower() for t in selected_tokens]
    labels: list[str] = []
    for _, row in frame.iterrows():
        track_toks = [t.lower() for t in parse_tags_cell(row.get("tags"))]
        hit = None
        for wanted in sel:
            if wanted in track_toks:
                hit = wanted
                break
        labels.append(hit if hit else "other / no selected sub-tag")
    return pd.Series(labels, index=frame.index)


def collapse_rare_categories(frame: pd.DataFrame, column: str, max_shown: int = 22) -> tuple[pd.DataFrame, str]:
    filled = frame[column].fillna("unknown").astype(str)
    vc = filled.value_counts()
    if len(vc) <= max_shown:
        return frame.copy(), column
    keep = set(vc.head(max_shown).index)
    out = frame.copy()
    plot_col = f"{column}_grouped"
    out[plot_col] = filled.where(filled.isin(keep), "other")
    return out, plot_col


def render_topology(tracks: pd.DataFrame, engine: RetrievalEngine) -> None:
    st.markdown("## Latent Topology")
    st.markdown(
        """
        <div class="section-intro">
        Global PCA axes are a <b>lossy 3D lens</b> on the 64-D ACE-Step VAE latents. The full
        1,734-point cloud looks noisy when colored by broad proxy buckets; that overlap is expected.
        Use the control column to trim outliers, slice Jamendo sub-tags from the <code>tags</code>
        array, or re-color by artists that appear a bounded number of times. K-means clusters summarize
        coarse acoustic neighborhoods for evaluation and overlays — micro-cluster mode increases resolution for exploration.
        </div>
        """,
        unsafe_allow_html=True,
    )
    ctrl, main = st.columns([3, 7])
    with ctrl:
        st.markdown("### Filters & coloring")
        st.caption(
            "Leave refinement off for the deliberately “messy” global view. Turn it on to carve "
            "finer neighborhoods for screenshots or intuition."
        )
        use_refine = st.toggle(
            "Refine subset (sub-tags, buckets, outlier trims)",
            value=False,
            key="topology_refine",
        )
        color_by = st.radio(
            "Color by",
            [
                "primary_tag",
                "cluster",
                "micro_cluster",
                "subgenre_tokens",
                "artist_frequency",
            ],
            format_func={
                "primary_tag": "Proxy bucket",
                "cluster": "Stored K-Means (10)",
                "micro_cluster": "120 micro-clusters",
                "subgenre_tokens": "Jamendo sub-tags (pick tokens)",
                "artist_frequency": "Artist (frequency band)",
            }.get,
            key="topology_color_mode",
        )
        micro_k = 120
        max_legend_artists = st.slider("Max legend groups (artist / sub-tag)", 12, 36, 22, key="topology_legend_cap")

        proxy_pick = st.multiselect(
            "Limit to proxy buckets (optional)",
            sorted(tracks["primary_tag"].dropna().unique()),
            key="topology_proxy_pick",
        )
        vocab = jamendo_subtag_vocabulary(tracks, max_tokens=80)
        subtag_pick = st.multiselect(
            "Jamendo sub-tags (from tags array)",
            vocab,
            key="topology_subtag_pick",
        )
        substr = st.text_input("Substring within tags (optional)", key="topology_substr")

        st.markdown("**Outlier thresholds**")
        trim_pct = st.slider("Drop farthest % from PCA centroid", 0, 35, 0, key="topology_centroid_trim")
        q_low = st.slider("Per-axis quantile low keep", 0.0, 0.2, 0.01, 0.005, key="topology_qlow")
        q_high = st.slider("Per-axis quantile high keep", 0.8, 1.0, 0.99, 0.005, key="topology_qhigh")

        p_min = st.slider("Artist min tracks (in filtered frame)", 1, 12, 2, key="topology_pmin")
        p_max = st.slider("Artist max tracks (in filtered frame)", 2, 30, 8, key="topology_pmax")

    plot_base = tracks.copy()
    if use_refine:
        refined = refine_topology_frame(plot_base, proxy_pick or None, subtag_pick, substr)
        refined = apply_axis_quantile_trim(refined, q_low, q_high)
        refined = apply_centroid_distance_trim(refined, float(trim_pct))
    else:
        refined = plot_base
        if proxy_pick:
            refined = refined[refined["primary_tag"].isin(proxy_pick)].copy()

    if refined.empty:
        st.warning("No tracks left after filters — relax thresholds or bucket choices.")
        return

    plot_frame = refined
    color_col = color_by

    if color_by == "micro_cluster":
        sub_embeddings = engine.embeddings[plot_frame["faiss_id"].astype(int).values]
        n = len(sub_embeddings)
        if n < 3:
            plot_frame = plot_frame.copy()
            color_col = "primary_tag"
        else:
            k = min(micro_k, n)
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            plot_frame = plot_frame.copy()
            plot_frame["micro_cluster"] = model.fit_predict(np.asarray(sub_embeddings, dtype="float32")).astype(str)
            color_col = "micro_cluster"
    elif color_by == "cluster":
        plot_frame = plot_frame.copy()
        plot_frame["cluster"] = plot_frame["cluster"].astype(str)
        color_col = "cluster"
    elif color_by == "subgenre_tokens":
        if not subtag_pick:
            st.warning("Pick at least one Jamendo sub-tag to color by this mode.")
            color_col = "primary_tag"
        else:
            plot_frame = plot_frame.copy()
            plot_frame["subgenre_color"] = subgenre_color_labels(plot_frame, subtag_pick)
            plot_frame, color_col = collapse_rare_categories(plot_frame, "subgenre_color", max_legend_artists)
    elif color_by == "artist_frequency":
        counts = plot_frame["artist_display_name"].fillna("Unknown").value_counts()
        lo, hi = min(p_min, p_max), max(p_min, p_max)
        allowed = counts[(counts >= lo) & (counts <= hi)].index
        plot_frame = plot_frame[plot_frame["artist_display_name"].isin(allowed)].copy()
        if plot_frame.empty:
            st.warning("No artists in that frequency band for the current frame. Widen P/Q or filters.")
            return
        plot_frame, color_col = collapse_rare_categories(plot_frame, "artist_display_name", max_legend_artists)
    else:
        plot_frame = plot_frame.copy()
        color_col = "primary_tag"

    with ctrl:
        st.metric("Points in plot", len(plot_frame))
        with st.expander("Jamendo tag vocabulary snapshot"):
            counts = Counter()
            for cell in tracks["tags"].tolist():
                for tok in parse_tags_cell(cell):
                    counts[str(tok).lower()] += 1
            st.caption(f"{len(counts)} distinct tokens across the corpus (top 25 by frequency).")
            st.dataframe(
                pd.DataFrame(counts.most_common(25), columns=["token", "track_count"]),
                hide_index=True,
                height=260,
            )

    with main:
        st.markdown("### Global PCA projection")
        info_card(
            "PCA vs. retrieval",
            "Nearest-neighbor search uses full 64-D latents (or ablations like PCA10). This page only visualizes "
            "the first three principal axes stored per track so we can rotate a faithful 3D summary of the same vectors.",
        )
        info_card(
            "Why keep K-means?",
            "Ten stored clusters give a coarse unsupervised partition of the VAE geometry — useful for quick overlays, "
            "baseline topology metrics (silhouette / DBI), and comparing against micro-clusters. They are not genre labels; "
            "they approximate contiguous regions in latent space that often align with local kNN neighborhoods.",
        )
        fig = topology_plot(plot_frame, color_column=color_col, height=720)
        st.plotly_chart(fig, width="stretch")


def format_track(row: pd.Series) -> str:
    tag = row.get("primary_tag", "untagged")
    extra = format_tags_display(row.get("tags"), max_chars=160)
    if extra:
        return f"{display_title(row)} - {display_artist(row)} [{tag}] — tags: {extra}"
    return f"{display_title(row)} - {display_artist(row)} [{tag}]"


def preview_path(paths: ArtifactPaths, row: pd.Series) -> Path | None:
    preview = row.get("preview_path")
    if not preview or pd.isna(preview):
        return None
    path = Path(str(preview))
    return path if path.is_absolute() else paths.root / path


def audio_preview(paths: ArtifactPaths, row: pd.Series) -> None:
    path = preview_path(paths, row)
    if path and path.exists():
        st.audio(str(path))
    else:
        st.caption("No preview snippet available.")


def available_plot_frame(tracks: pd.DataFrame, engine: RetrievalEngine) -> pd.DataFrame:
    frame = tracks.copy()
    columns = ["pca_1", "pca_2", "pca_3"]
    if not all(column in frame.columns for column in columns) or frame[columns].isna().any().any():
        vectors = engine.embeddings[:, :3]
        frame["pca_1"] = vectors[:, 0]
        frame["pca_2"] = vectors[:, 1]
        frame["pca_3"] = vectors[:, 2]
    return frame


def metric_card(label: str, value: str | int | float, note: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def info_card(title: str, copy: str) -> None:
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">{title}</div>
            <div class="card-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">ACE-Step 1.5 VAE latent recommendation</div>
            <div class="hero-title">A content-first recommender that starts from the audio itself.</div>
            <div class="hero-copy">
                Instead of learning from user histories or popularity signals, this dashboard
                builds synthetic listening contexts from raw-audio latent geometry and asks:
                which songs complete the same acoustic neighborhood?
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_pipeline_diagram() -> None:
    if PIPELINE_DIAGRAM_PATH.exists():
        st.image(
            str(PIPELINE_DIAGRAM_PATH),
            width="stretch",
        )
        st.caption(
            "Poster-friendly architecture diagram. SVG source lives next to this PNG in `documentations/figures`."
        )
    else:
        st.warning(
            "Pipeline diagram asset is missing. Regenerate it with "
            "`python scripts/generate_pipeline_architecture_diagram.py`."
        )


def render_bucket_chart(tracks: pd.DataFrame) -> None:
    bucket_counts = (
        tracks["primary_tag"]
        .value_counts()
        .rename_axis("proxy_bucket")
        .reset_index(name="tracks")
    )
    fig = px.bar(
        bucket_counts,
        x="tracks",
        y="proxy_bucket",
        orientation="h",
        color="tracks",
        color_continuous_scale=["#313849", "#7cc7ff", "#ff4b5c"],
        text="tracks",
        height=430,
    )
    fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f4f6fb"},
        coloraxis_showscale=False,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")


def render_track_table(frame: pd.DataFrame, height: int = 380) -> None:
    frame = ensure_tags_display(frame)
    columns = [
        "faiss_id",
        "track_id",
        "track_display_title",
        "artist_display_name",
        "album_display_title",
        "primary_tag",
        "tags_display",
        "duration",
        "cluster",
    ]
    shown = [column for column in columns if column in frame.columns]
    st.dataframe(frame[shown], width="stretch", height=height, hide_index=True)


def filter_tracks(
    tracks: pd.DataFrame,
    key_prefix: str,
    include_bucket: bool = True,
    include_artist: bool = True,
    include_album: bool = True,
) -> pd.DataFrame:
    filtered = tracks.copy()
    left, middle, right = st.columns(3)
    if include_bucket:
        bucket_values = sorted(filtered["primary_tag"].dropna().unique())
        selected_buckets = left.multiselect("Proxy bucket", bucket_values, key=f"{key_prefix}_bucket")
        if selected_buckets:
            filtered = filtered[filtered["primary_tag"].isin(selected_buckets)]
    if include_artist and "artist_display_name" in filtered:
        artist_values = sorted(filtered["artist_display_name"].dropna().unique())
        selected_artists = middle.multiselect("Artist", artist_values, key=f"{key_prefix}_artist")
        if selected_artists:
            filtered = filtered[filtered["artist_display_name"].isin(selected_artists)]
    if include_album and "album_display_title" in filtered:
        album_values = sorted(filtered["album_display_title"].dropna().unique())
        selected_albums = right.multiselect("Album", album_values, key=f"{key_prefix}_album")
        if selected_albums:
            filtered = filtered[filtered["album_display_title"].isin(selected_albums)]
    search = st.text_input("Search track, artist, album, tag, or sub-tag", key=f"{key_prefix}_search")
    if search:
        filtered = ensure_tags_display(filtered)
        haystack = (
            filtered["track_display_title"].fillna("")
            + " "
            + filtered["artist_display_name"].fillna("")
            + " "
            + filtered["album_display_title"].fillna("")
            + " "
            + filtered["primary_tag"].fillna("")
            + " "
            + filtered["tags_display"].fillna("")
        ).str.lower()
        filtered = filtered[haystack.str.contains(search.lower(), regex=False)]
    return filtered


def topology_plot(
    tracks: pd.DataFrame,
    highlight_ids: set[int] | None = None,
    neighbor_ids: set[int] | None = None,
    color_column: str = "primary_tag",
    height: int = 620,
) -> go.Figure:
    highlight_ids = highlight_ids or set()
    neighbor_ids = neighbor_ids or set()
    frame = ensure_tags_display(tracks.copy())
    frame["role"] = "corpus"
    frame.loc[frame["faiss_id"].isin(neighbor_ids), "role"] = "recommendation"
    frame.loc[frame["faiss_id"].isin(highlight_ids), "role"] = "seed"
    frame["track"] = frame.apply(display_title, axis=1)
    frame["artist_name"] = frame.apply(display_artist, axis=1)
    hover = [c for c in ["faiss_id", "track", "artist_name", "primary_tag", "tags_display", "cluster"] if c in frame.columns]
    fig = px.scatter_3d(
        frame,
        x="pca_1",
        y="pca_2",
        z="pca_3",
        color=color_column,
        symbol="role",
        hover_data=hover,
        height=height,
        opacity=0.78,
    )
    fig.update_traces(marker={"size": 4})
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f4f6fb"},
        legend={"itemsizing": "constant"},
        margin={"l": 0, "r": 0, "t": 20, "b": 0},
        scene={
            "xaxis": {"backgroundcolor": "#0f1117", "gridcolor": "rgba(255,255,255,0.08)"},
            "yaxis": {"backgroundcolor": "#0f1117", "gridcolor": "rgba(255,255,255,0.08)"},
            "zaxis": {"backgroundcolor": "#0f1117", "gridcolor": "rgba(255,255,255,0.08)"},
        },
    )
    return fig


def playlist_summary(playlists: pd.DataFrame, playlist_tracks: pd.DataFrame) -> pd.DataFrame:
    if playlists.empty or playlist_tracks.empty:
        return pd.DataFrame()
    tag_summary = (
        playlist_tracks.groupby("playlist_id")["primary_tag"]
        .agg(lambda values: ", ".join(values.value_counts().head(3).index))
        .rename("dominant_buckets")
    )
    artist_summary = (
        playlist_tracks.groupby("playlist_id")["artist_display_name"]
        .nunique()
        .rename("unique_artists")
    )
    return (
        playlists.merge(tag_summary, left_on="playlist_id", right_index=True, how="left")
        .merge(artist_summary, left_on="playlist_id", right_index=True, how="left")
        .sort_values("playlist_id")
    )


def labels_for_tracks(frame: pd.DataFrame) -> dict[str, int]:
    frame = ensure_tags_display(frame)
    safe = frame.sort_values(["artist_display_name", "album_display_title", "track_display_title", "faiss_id"])
    labels: dict[str, int] = {}
    for _, row in safe.iterrows():
        tag = row["primary_tag"]
        td = format_tags_display(row.get("tags"), max_chars=80)
        suffix = f" | sub-tags: {td}" if td else ""
        labels[f"{display_title(row)} | {display_artist(row)} | {display_album(row)} | {tag}{suffix}"] = int(row["faiss_id"])
    return labels


def render_strategy_cards() -> None:
    cols = st.columns(4)
    cards = [
        ("1. Anchor", "Randomly choose a track that has not exceeded the reuse cap."),
        ("2. Expand", "Fill the playlist with nearest latent neighbors from raw 64D cosine search."),
        ("3. Control Reuse", "No song can dominate the synthetic set; reuse is capped across playlists."),
        ("4. Complete", "Hold out playlist tracks, average the remaining seeds, and retrieve completions."),
    ]
    for col, (title, copy) in zip(cols, cards):
        with col:
            info_card(title, copy)


def render_mode_cards() -> None:
    cols = st.columns(len(RETRIEVAL_MODES))
    for col, mode in zip(cols, RETRIEVAL_MODES.values()):
        with col:
            info_card(mode["label"], f"{mode['use']}<br><br><b>Distance:</b> {mode['distance']}")


def render_overview(metrics: dict, tracks: pd.DataFrame, playlists: pd.DataFrame, playlist_tracks: pd.DataFrame) -> None:
    render_hero()
    dataset = metrics.get("dataset", {})
    playlist_metrics = metrics.get("playlist_completion", {})
    cols = st.columns(5)
    with cols[0]:
        metric_card("Tracks", f"{len(tracks):,}", "Unique merged audio items")
    with cols[1]:
        metric_card("Proxy Buckets", tracks["primary_tag"].nunique(), "Weak style/acoustic labels")
    with cols[2]:
        metric_card("Embedding Dim", dataset.get("embedding_dimensions", 64), "ACE-Step VAE latent vector")
    with cols[3]:
        metric_card("Playlists", len(playlists), "Procedurally generated")
    with cols[4]:
        metric_card("Completion HitRate@4", f"{playlist_metrics.get('hit_rate@4', 0):.3f}", "Playlist holdout success")

    st.markdown("### Pipeline Architecture")
    st.markdown(
        """
        <div class="section-intro">
        The dashboard is organized around the full project loop: sample open music,
        encode it with the ACE-Step 1.5 VAE, build latent neighborhoods, generate
        synthetic playlists, and test whether centroid queries can complete those playlists.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_pipeline_diagram()

    cols = st.columns(3)
    with cols[0]:
        info_card(
            "Why ACE-Step VAE?",
            "The VAE gives a compact audio representation without loading the full music generation stack. Each song becomes one normalized 64-D acoustic vector.",
        )
    with cols[1]:
        info_card(
            "Why procedural playlists?",
            "For this proof of concept, playlists are constructed directly from latent neighborhoods. That makes the completion workflow easy to inspect and demo.",
        )
    with cols[2]:
        info_card(
            "What gets evaluated?",
            "The main story is playlist completion: hold out tracks, average the remaining seeds, and measure whether the latent search recovers the missing songs.",
        )

    st.markdown("### Artifact Health")
    integrity = metrics.get("artifact_integrity", {})
    health_cols = st.columns(4)
    with health_cols[0]:
        metric_card("Embeddings", str(integrity.get("embedding_shape", [len(tracks), 64])), "Expected shape is N x 64")
    with health_cols[1]:
        metric_card("Finite Values", "Pass" if integrity.get("embeddings_finite", False) else "Check", "No NaN/Inf in latent matrix")
    with health_cols[2]:
        metric_card("Index Alignment", "Pass" if integrity.get("faiss_ids_contiguous", False) else "Check", "SQLite rows align with vector rows")
    with health_cols[3]:
        metric_card("Preview Files", f"{int(tracks['preview_path'].notna().sum()):,}", "15 second MP3 snippets")

    if not playlist_tracks.empty:
        st.markdown("### Playlist Corpus At A Glance")
        playlist_sizes = playlist_tracks.groupby("playlist_id").size()
        size_frame = playlist_sizes.value_counts().rename_axis("playlist_size").reset_index(name="count")
        fig = px.bar(
            size_frame.sort_values("playlist_size"),
            x="playlist_size",
            y="count",
            text="count",
            color="count",
            color_continuous_scale=["#313849", "#ff4b5c"],
            height=280,
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#f4f6fb"},
            coloraxis_showscale=False,
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
        )
        st.plotly_chart(fig, width="stretch")


def render_dataset(tracks: pd.DataFrame) -> None:
    st.markdown("## Dataset Explorer")
    st.markdown(
        """
        <div class="section-intro">
        The dataset is a tag-balanced MTG-Jamendo sample. Buckets are broad proxy
        listening contexts, not hard genres: they give us a way to sample and
        evaluate acoustic neighborhoods.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1])
    with left:
        render_bucket_chart(tracks)
    with right:
        counts = tracks["primary_tag"].value_counts()
        metric_cols = st.columns(2)
        with metric_cols[0]:
            metric_card("Largest Bucket", f"{counts.max():,}", counts.idxmax())
        with metric_cols[1]:
            metric_card("Smallest Bucket", f"{counts.min():,}", counts.idxmin())
        info_card(
            "Display Metadata",
            "MTG-Jamendo artifacts provide stable track, artist, and album IDs. The UI uses readable Jamendo fallback labels so tables and filters remain interpretable.",
        )

    st.markdown("### Search Tracks")
    filtered = filter_tracks(tracks, "dataset")
    render_track_table(filtered, height=470)


def render_playlists(
    paths: ArtifactPaths,
    engine: RetrievalEngine,
    tracks: pd.DataFrame,
    playlists: pd.DataFrame,
    playlist_tracks: pd.DataFrame,
) -> None:
    st.markdown("## Procedurally Generated Playlists")
    st.markdown(
        """
        <div class="section-intro">
        This proof of concept frames recommendation around synthetic acoustic playlists:
        random anchors are expanded with nearest latent neighbors, then playlist completion
        is tested by holding out tracks and querying with the centroid of the remaining songs.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_strategy_cards()

    if playlists.empty or playlist_tracks.empty:
        st.info("Generated playlists are not available yet. Run `scripts/evaluate_artifacts.py` to create them.")
        return

    summary = playlist_summary(playlists, playlist_tracks)
    st.markdown("### Playlist Index")
    render_cols = st.columns(4)
    sizes = playlist_tracks.groupby("playlist_id").size()
    with render_cols[0]:
        metric_card("Generated Playlists", len(playlists), "Stored in SQLite")
    with render_cols[1]:
        metric_card("Tracks In Playlists", len(playlist_tracks), "Total ordered entries")
    with render_cols[2]:
        metric_card("Size Range", f"{sizes.min()}-{sizes.max()}", "Songs per playlist")
    with render_cols[3]:
        metric_card("Unique Playlist Songs", playlist_tracks["faiss_id"].nunique(), "After reuse cap")

    st.dataframe(
        summary[["playlist_id", "strategy", "size", "anchor_faiss_id", "dominant_buckets", "unique_artists"]],
        width="stretch",
        height=260,
        hide_index=True,
    )

    st.markdown("### Explore A Playlist")
    labels = {
        f"{row['playlist_id']} | {int(row['size'])} tracks | {row['dominant_buckets']}": row["playlist_id"]
        for _, row in summary.iterrows()
    }
    selected_label = st.selectbox("Playlist", list(labels), key="playlist_select")
    playlist_id = labels[selected_label]
    selected_tracks = playlist_tracks[playlist_tracks["playlist_id"] == playlist_id].sort_values("position")

    filter_left, filter_right = st.columns(2)
    artist_filter = filter_left.multiselect(
        "Artist filter inside playlist",
        sorted(selected_tracks["artist_display_name"].dropna().unique()),
        key="playlist_artist_filter",
    )
    album_filter = filter_right.multiselect(
        "Album filter inside playlist",
        sorted(selected_tracks["album_display_title"].dropna().unique()),
        key="playlist_album_filter",
    )
    playlist_view = selected_tracks
    if artist_filter:
        playlist_view = playlist_view[playlist_view["artist_display_name"].isin(artist_filter)]
    if album_filter:
        playlist_view = playlist_view[playlist_view["album_display_title"].isin(album_filter)]

    left, right = st.columns([1.15, 1])
    with left:
        st.dataframe(
            playlist_view[
                [
                    "position",
                    "faiss_id",
                    "track_display_title",
                    "artist_display_name",
                    "album_display_title",
                    "primary_tag",
                    "tags_display",
                    "duration",
                ]
            ],
            width="stretch",
            height=360,
            hide_index=True,
        )
        if not playlist_view.empty:
            preview_labels = {
                f"{int(row.position)}. {row.track_display_title} - {row.artist_display_name}": row
                for _, row in playlist_view.iterrows()
            }
            selected_preview = st.selectbox("Preview playlist track", list(preview_labels), key="playlist_preview")
            audio_preview(paths, pd.Series(preview_labels[selected_preview]))
    with right:
        st.markdown("#### Local Playlist Topology")
        playlist_ids = set(selected_tracks["faiss_id"].astype(int))
        surrounding_ids = set()
        for seed_id in list(playlist_ids)[:3]:
            neighbors = engine.query(seed_id, k=8, mode="raw64")
            surrounding_ids.update(neighbors["faiss_id"].astype(int).tolist())
        local_ids = playlist_ids | surrounding_ids
        local_frame = tracks[tracks["faiss_id"].isin(local_ids)]
        fig = topology_plot(
            local_frame,
            highlight_ids=playlist_ids,
            neighbor_ids=surrounding_ids - playlist_ids,
            color_column="role",
            height=470,
        )
        st.plotly_chart(fig, width="stretch")


def render_recommend(
    paths: ArtifactPaths,
    engine: RetrievalEngine,
    tracks: pd.DataFrame,
    playlists: pd.DataFrame,
    playlist_tracks: pd.DataFrame,
) -> None:
    st.markdown("## Recommendation Workbench")
    st.markdown(
        """
        <div class="section-intro">
        Use this page in two ways: build a seed set manually from artists/albums/tracks,
        or start from a generated playlist and ask the latent space to complete it.
        Multi-seed recommendations average selected embeddings into a centroid query.
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_mode_cards()

    manual_tab, playlist_tab = st.tabs(["Seed-Based Recommendation", "Playlist Completion Demo"])

    with manual_tab:
        mode = st.radio(
            "Similarity strategy",
            list(RETRIEVAL_MODES),
            horizontal=True,
            format_func=lambda mode_name: RETRIEVAL_MODES[mode_name]["label"],
            key="manual_mode",
        )
        k = st.slider("Number of recommendations", 4, 20, 8, key="manual_k")
        st.markdown("### Find Seed Tracks")
        filtered = filter_tracks(tracks, "recommend")
        labels = labels_for_tracks(filtered)
        if not labels:
            st.warning("No tracks match the current filters.")
            return
        selected = st.multiselect(
            "Select up to five seed tracks",
            list(labels),
            default=list(labels)[:1],
            max_selections=5,
            key="manual_seed_select",
        )
        if not selected:
            st.info("Select at least one seed track.")
            return
        seed_ids = [labels[label] for label in selected]
        seed_rows = tracks[tracks["faiss_id"].isin(seed_ids)]
        if len(seed_ids) == 1:
            neighbors = engine.query(seed_ids[0], k=k, mode=mode)
        else:
            neighbors = engine.query_many(seed_ids, k=k, mode=mode)
        render_recommendation_results(paths, tracks, seed_rows, neighbors, "manual_results")

    with playlist_tab:
        if playlists.empty or playlist_tracks.empty:
            st.info("Generated playlists are not available yet.")
            return
        summary = playlist_summary(playlists, playlist_tracks)
        labels = {
            f"{row['playlist_id']} | {int(row['size'])} tracks | {row['dominant_buckets']}": row["playlist_id"]
            for _, row in summary.iterrows()
        }
        selected_label = st.selectbox("Choose generated playlist", list(labels), key="completion_playlist")
        playlist_id = labels[selected_label]
        members = playlist_tracks[playlist_tracks["playlist_id"] == playlist_id].sort_values("position")
        holdout_count = st.slider("Hold out N centroid-central tracks", 1, min(3, max(1, len(members) - 2)), 2)
        k = st.slider("Completion candidates", 4, 12, 4, key="playlist_completion_k")
        query_ids, holdout_ids = split_playlist_holdout(
            engine,
            members["faiss_id"].astype(int).tolist(),
            holdout_count=holdout_count,
            mode="raw64",
            strategy="centroid",
        )
        query_members = members[members["faiss_id"].isin(query_ids)]
        holdouts = members[members["faiss_id"].isin(holdout_ids)]
        results = engine.query_many(query_ids, k=k, mode="raw64")
        hit_ids = set(results["faiss_id"].astype(int)) & set(holdout_ids)

        cols = st.columns(4)
        with cols[0]:
            metric_card("Query Seeds", len(query_members), "Centroid input")
        with cols[1]:
            metric_card("Held Out", len(holdouts), "Hidden target tracks")
        with cols[2]:
            metric_card("Hits In Top K", len(hit_ids), "Recovered holdouts")
        with cols[3]:
            metric_card("This Playlist Precision", f"{len(hit_ids) / max(1, k):.3f}", f"Selected playlist only, top {k}")

        st.caption(
            "These four cards are computed live for the selected playlist and slider settings. "
            "The Evaluation tab reports aggregate metrics across all generated playlists."
        )

        left, right = st.columns(2)
        with left:
            st.markdown("#### Playlist Seeds")
            render_track_table(query_members, height=300)
        with right:
            st.markdown("#### Held-Out Targets")
            render_track_table(holdouts, height=300)
        render_recommendation_results(paths, tracks, query_members, results, "playlist_completion_results", holdout_ids=hit_ids)


def render_recommendation_results(
    paths: ArtifactPaths,
    tracks: pd.DataFrame,
    seed_rows: pd.DataFrame,
    neighbors: pd.DataFrame,
    key_prefix: str,
    holdout_ids: set[int] | None = None,
) -> None:
    holdout_ids = holdout_ids or set()
    left, right = st.columns([0.85, 1.15])
    with left:
        st.markdown("### Seeds")
        for _, row in seed_rows.iterrows():
            st.write(format_track(row))
            audio_preview(paths, row)
    with right:
        st.markdown("### Recommendations")
        if neighbors.empty:
            st.warning("No recommendations found.")
        else:
            tracks_tagged = ensure_tags_display(tracks)
            tag_map = tracks_tagged.set_index("faiss_id")["tags_display"].to_dict()
            result = neighbors.copy()
            result["track"] = result.apply(display_title, axis=1)
            result["artist_name"] = result.apply(display_artist, axis=1)
            result["album_name"] = result.apply(display_album, axis=1)
            result["tags_display"] = result["faiss_id"].astype(int).map(lambda i: tag_map.get(i, ""))
            result["completion_hit"] = result["faiss_id"].astype(int).isin(holdout_ids)
            st.dataframe(
                result[
                    [
                        "rank",
                        "track",
                        "artist_name",
                        "album_name",
                        "primary_tag",
                        "tags_display",
                        "distance",
                        "completion_hit",
                    ]
                ],
                width="stretch",
                height=320,
                hide_index=True,
            )
            preview_label = st.selectbox(
                "Preview recommendation",
                result["rank"].astype(str) + ". " + result["track"],
                key=f"{key_prefix}_preview",
            )
            rank = int(preview_label.split(".", 1)[0])
            audio_preview(paths, result.loc[result["rank"] == rank].iloc[0])

    st.markdown("### Recommendation Geometry")
    local_plot_frame = (
        pd.concat([seed_rows, neighbors], ignore_index=True)
        .drop_duplicates("faiss_id")
        .reset_index(drop=True)
    )
    fig = topology_plot(
        local_plot_frame,
        highlight_ids=set(seed_rows["faiss_id"].astype(int)),
        neighbor_ids=set(neighbors["faiss_id"].astype(int)) if not neighbors.empty else set(),
        color_column="primary_tag",
        height=640,
    )
    fig.update_traces(marker={"size": 8, "opacity": 0.95})
    st.plotly_chart(fig, width="stretch")


def render_evaluation(metrics: dict) -> None:
    st.markdown("## Evaluation")
    st.markdown(
        """
        <div class="section-intro">
        The dashboard emphasizes playlist completion because that is the clearest
        proof-of-concept behavior: given several songs from a synthetic playlist,
        can the centroid query recover hidden playlist members?
        </div>
        """,
        unsafe_allow_html=True,
    )
    retrieval = metrics.get("retrieval", {})
    playlist = metrics.get("playlist_completion", {})
    generated = metrics.get("generated_playlists", {})
    topology = metrics.get("topology", {})
    ablation = metrics.get("ablation", {})

    cols = st.columns(4)
    with cols[0]:
        metric_card("Playlist HitRate@4", f"{playlist.get('hit_rate@4', 0):.3f}", "Fraction of playlists with at least one held-out hit")
    with cols[1]:
        metric_card("Playlist Precision@4", f"{playlist.get('precision@4', 0):.3f}", "How many top-4 candidates were held-out tracks")
    with cols[2]:
        metric_card("Playlist Recall@4", f"{playlist.get('recall@4', 0):.3f}", "How many held-out tracks were recovered")
    with cols[3]:
        metric_card("Playlist MRR@4", f"{playlist.get('mrr@4', 0):.3f}", "How early the first held-out hit appears")

    st.markdown("### Metric Guide")
    guide_cols = st.columns(3)
    with guide_cols[0]:
        info_card("Precision@K", "Of the K returned songs, the fraction that exactly match the hidden playlist tracks. Similar-but-not-held-out songs still count as misses.")
    with guide_cols[1]:
        info_card("HitRate@K", "A binary success measure: did any held-out playlist track appear in the top K candidates?")
    with guide_cols[2]:
        info_card("MRR", "Mean reciprocal rank rewards systems that place the first correct completion near the top. Higher means the first hit appears earlier.")

    st.markdown("### Playlist Protocol")
    protocol_cols = st.columns(3)
    with protocol_cols[0]:
        info_card(
            "Generation Strategy",
            "Playlists are centroid-cohesive acoustic neighborhoods: random anchors are expanded with nearby raw-64D VAE neighbors, then tightened around the local centroid.",
        )
    with protocol_cols[1]:
        info_card(
            "Holdout Strategy",
            "Evaluation hides centroid-central playlist members instead of the least-similar tail tracks. This better matches the POC goal: recover missing songs from the same acoustic pocket.",
        )
    with protocol_cols[2]:
        info_card(
            "Why Not Perfect?",
            "The search runs against all 1,734 songs. If another acoustically similar song ranks above the exact hidden track, that is useful recommendation behavior but still a metric miss.",
        )

    st.markdown("### Playlist Completion Summary")
    playlist_frame = pd.DataFrame(
        [
            {"metric": "HitRate@4", "value": playlist.get("hit_rate@4", 0)},
            {"metric": "Precision@4", "value": playlist.get("precision@4", 0)},
            {"metric": "Recall@4", "value": playlist.get("recall@4", 0)},
            {"metric": "MRR@4", "value": playlist.get("mrr@4", 0)},
        ]
    )
    fig = px.bar(
        playlist_frame,
        x="metric",
        y="value",
        text=playlist_frame["value"].map(lambda value: f"{value:.3f}"),
        color="value",
        color_continuous_scale=["#313849", "#7cc7ff", "#ff4b5c"],
        range_y=[0, 1],
        height=360,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#f4f6fb"},
        coloraxis_showscale=False,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("### Retrieval And Ablation Context")
    st.markdown(
        """
        <div class="section-intro">
        The chart on the left is proxy-bucket retrieval: for every seed track, we ask whether
        its nearest neighbors share the same metadata-derived bucket. The table on the right
        is an ablation over retrieval spaces. These numbers are not playlist-completion scores;
        they show how much information is preserved when raw 64-D VAE vectors are compressed
        or reweighted.
        </div>
        """,
        unsafe_allow_html=True,
    )
    left, right = st.columns([1, 1])
    with left:
        precision_by_tag = retrieval.get("precision_by_tag", {})
        if precision_by_tag:
            precision_frame = pd.DataFrame(
                [{"bucket": key, "precision": value} for key, value in precision_by_tag.items()]
            )
            fig = px.bar(
                precision_frame.sort_values("precision"),
                x="precision",
                y="bucket",
                orientation="h",
                text=precision_frame.sort_values("precision")["precision"].map(lambda value: f"{value:.3f}"),
                color="precision",
                color_continuous_scale=["#313849", "#7cc7ff"],
                height=430,
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font={"color": "#f4f6fb"},
                coloraxis_showscale=False,
                margin={"l": 10, "r": 10, "t": 10, "b": 10},
            )
            st.plotly_chart(fig, width="stretch")
    with right:
        ablation_rows = []
        for name, values in ablation.items():
            ablation_rows.append(
                {
                    "Retrieval space": RETRIEVAL_MODES.get(name, {"label": name})["label"],
                    "What it tests": RETRIEVAL_MODES.get(name, {"use": "Alternative retrieval space."})["use"],
                    "Distance": RETRIEVAL_MODES.get(name, {"distance": "N/A"})["distance"],
                    "Proxy Precision@10": values.get("precision_at_10", 0),
                }
            )
        if ablation_rows:
            ablation_frame = pd.DataFrame(ablation_rows).sort_values("Proxy Precision@10", ascending=False)
            st.dataframe(ablation_frame, width="stretch", hide_index=True)
            best = ablation_frame.iloc[0]
            info_card(
                "How To Read This Table",
                f"`{best['Retrieval space']}` is strongest here because it keeps the full VAE latent signal. "
                "Lower PCA scores mean the reduced spaces are useful for visualization but discard recommendation-relevant detail.",
            )
        centroid = topology.get("centroid", {})
        clustering = topology.get("clustering", {})
        info_card(
            "Topology Signals",
            f"Triplet success: {topology.get('triplet_success_rate', 0):.3f}<br>"
            f"Centroid separation ratio: {centroid.get('separation_ratio', 0):.3f}<br>"
            f"Silhouette: {clustering.get('silhouette', 0):.3f}<br>"
            f"Playlist strategy: {generated.get('strategy', 'N/A')}<br>"
            f"Holdout strategy: {generated.get('holdout_strategy', 'N/A')}",
        )


apply_theme()
paths, engine, tracks, metrics, has_artifacts = load_bundle()
tracks = available_plot_frame(tracks, engine)
tracks = ensure_tags_display(tracks)
engine.tracks = tracks
playlists, playlist_tracks = load_playlist_tables(paths.metadata_path)
playlist_tracks = ensure_tags_display(playlist_tracks)

st.title("latent-recommend")
st.caption("Content-first music recommendation using ACE-Step 1.5 VAE latent geometry.")

if not has_artifacts:
    st.info("Running with demo data. Set `LATENT_RECOMMEND_ARTIFACTS` or create `artifacts/merged` for the final bundle.")

tab_overview, tab_dataset, tab_playlists, tab_recommend, tab_topology, tab_eval = st.tabs(
    ["Overview", "Dataset", "Playlists", "Recommend", "Latent Topology", "Evaluation"]
)

with tab_overview:
    render_overview(metrics, tracks, playlists, playlist_tracks)

with tab_dataset:
    render_dataset(tracks)

with tab_playlists:
    render_playlists(paths, engine, tracks, playlists, playlist_tracks)

with tab_recommend:
    render_recommend(paths, engine, tracks, playlists, playlist_tracks)

with tab_topology:
    render_topology(tracks, engine)

with tab_eval:
    render_evaluation(metrics)
