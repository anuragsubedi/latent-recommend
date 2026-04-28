"""Streamlit dashboard for latent-recommend Milestone 2."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from latent_recommend.artifacts import (
    artifacts_available,
    demo_embeddings,
    load_embeddings,
    load_metadata,
    load_metrics,
)
from latent_recommend.config import ArtifactPaths
from latent_recommend.retrieval import RetrievalEngine


st.set_page_config(
    page_title="latent-recommend",
    page_icon="music",
    layout="wide",
)


@st.cache_resource(show_spinner=False)
def load_engine() -> tuple[RetrievalEngine, pd.DataFrame, dict, bool]:
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
    return engine, tracks, load_metrics(paths), has_artifacts


def format_track(row: pd.Series) -> str:
    title = row.get("title") or row.get("track_id")
    artist = row.get("artist") or "Unknown artist"
    tag = row.get("primary_tag") or "untagged"
    return f"{title} - {artist} [{tag}]"


def audio_preview(row: pd.Series) -> None:
    preview_path = row.get("preview_path")
    if not preview_path or pd.isna(preview_path):
        st.caption("No preview snippet available for this track yet.")
        return
    path = Path(preview_path)
    if not path.is_absolute():
        path = ArtifactPaths().root / path
    if path.exists():
        st.audio(str(path))
    else:
        st.caption(f"Preview missing: `{preview_path}`")


def ensure_plot_coordinates(frame: pd.DataFrame, engine: RetrievalEngine) -> pd.DataFrame:
    columns = ["pca_1", "pca_2", "pca_3"]
    if all(column in frame.columns for column in columns) and not frame[columns].isna().any().any():
        return frame
    fallback = frame.copy()
    vectors = engine.embeddings[:, :3]
    fallback["pca_1"] = vectors[:, 0]
    fallback["pca_2"] = vectors[:, 1]
    fallback["pca_3"] = vectors[:, 2]
    return fallback


engine, tracks, metrics, has_artifacts = load_engine()
tracks = ensure_plot_coordinates(tracks, engine)
engine.tracks = tracks

st.title("latent-recommend")
st.caption(
    "Content-first music recommendation using ACE-Step 1.5 VAE latent geometry."
)

if not has_artifacts:
    st.info(
        "Final artifacts are not present yet, so the dashboard is running with a "
        "small demo fixture. Generate `artifacts/` from the Colab pipeline to see "
        "real recommendations."
    )

with st.sidebar:
    st.header("Recommendation Controls")
    mode = st.radio(
        "Retrieval space",
        ["faiss", "raw64", "pca10", "pca3"],
        format_func={
            "faiss": "Raw 64D FAISS / cosine",
            "raw64": "Raw 64D NumPy",
            "pca10": "10D PCA ablation",
            "pca3": "3D PCA ablation",
        }.get,
    )
    k = st.slider("Neighbors", min_value=3, max_value=25, value=10)
    tag_filter = st.multiselect(
        "Filter seed list by tag",
        sorted(tracks["primary_tag"].dropna().unique()),
    )

seed_pool = tracks
if tag_filter:
    seed_pool = seed_pool[seed_pool["primary_tag"].isin(tag_filter)]

seed_labels = {
    format_track(row): int(row["faiss_id"])
    for _, row in seed_pool.sort_values(["primary_tag", "title"]).iterrows()
}

if not seed_labels:
    st.warning("No tracks match the selected seed filters.")
    st.stop()

selected_label = st.sidebar.selectbox("Seed track", list(seed_labels))
seed_id = seed_labels[selected_label]
seed_row = tracks.loc[tracks["faiss_id"] == seed_id].iloc[0]
neighbors = engine.query(seed_id, k=k, mode=mode)

tab_recommend, tab_explorer, tab_manifold, tab_metrics = st.tabs(
    ["Recommendations", "Dataset Explorer", "Acoustic Manifold", "Evaluation"]
)

with tab_recommend:
    left, right = st.columns([1, 2])
    with left:
        st.subheader("Seed")
        st.write(format_track(seed_row))
        audio_preview(seed_row)
    with right:
        st.subheader("Nearest Neighbors")
        if neighbors.empty:
            st.warning("No neighbors found for this seed.")
        else:
            view = neighbors[
                ["rank", "title", "artist", "primary_tag", "distance", "preview_path"]
            ].copy()
            st.dataframe(view, use_container_width=True, hide_index=True)

            selected_neighbor = st.selectbox(
                "Preview a recommendation",
                neighbors["rank"].astype(str) + ". " + neighbors["title"].fillna("Untitled"),
            )
            rank = int(selected_neighbor.split(".", 1)[0])
            audio_preview(neighbors.loc[neighbors["rank"] == rank].iloc[0])

with tab_explorer:
    st.subheader("Proxy Playlists")
    counts = tracks["primary_tag"].value_counts().rename_axis("tag").reset_index(name="tracks")
    st.dataframe(counts, use_container_width=True, hide_index=True)

    explorer_tags = st.multiselect(
        "Show tracks for tags",
        sorted(tracks["primary_tag"].dropna().unique()),
        default=sorted(tracks["primary_tag"].dropna().unique())[:2],
        key="explorer_tags",
    )
    explorer = tracks
    if explorer_tags:
        explorer = explorer[explorer["primary_tag"].isin(explorer_tags)]
    st.dataframe(
        explorer[["faiss_id", "title", "artist", "primary_tag", "duration", "cluster"]],
        use_container_width=True,
        hide_index=True,
    )

with tab_manifold:
    st.subheader("3D Acoustic Topology")
    plot_frame = tracks.copy()
    plot_frame["role"] = "corpus"
    plot_frame.loc[plot_frame["faiss_id"] == seed_id, "role"] = "seed"
    plot_frame.loc[plot_frame["faiss_id"].isin(neighbors["faiss_id"]), "role"] = "neighbor"
    fig = px.scatter_3d(
        plot_frame,
        x="pca_1",
        y="pca_2",
        z="pca_3",
        color="primary_tag",
        symbol="role",
        hover_data=["title", "artist", "faiss_id", "cluster"],
        height=720,
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_metrics:
    st.subheader("Evaluation Summary")
    if not metrics:
        st.info("Metrics will appear here after `scripts/evaluate_artifacts.py` runs.")
    else:
        col1, col2, col3 = st.columns(3)
        retrieval = metrics.get("retrieval", {})
        topology = metrics.get("topology", {})
        col1.metric("Precision@10", f"{retrieval.get('precision_at_10', 0):.3f}")
        col2.metric("MRR@10", f"{retrieval.get('mrr_at_10', 0):.3f}")
        col3.metric("Triplet Success", f"{topology.get('triplet_success_rate', 0):.3f}")
        st.json(metrics)
