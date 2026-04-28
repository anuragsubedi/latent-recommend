# `latent-recommend`

`latent-recommend` is a content-first music recommendation project for a
Statistical Machine Learning final milestone. The final system uses 64-D latent
embeddings from the ACE-Step 1.5 waveform VAE to recommend songs by acoustic
geometry rather than user-history or popularity signals.

## Current Layout

```text
latent-recommend/
├── app/                         # Streamlit dashboard entrypoint
├── artifacts/                   # Generated M2 outputs; mostly ignored by git
├── documentations/              # Project notes, requirements, handoff docs
├── latent_recommend/            # Reusable M2 Python package
├── milestone1/                  # Archived Spotify-feature baseline
├── notebooks/                   # Colab extraction notebooks
├── scripts/                     # Local evaluation and validation commands
├── ACE-Step/                    # Cloned upstream model repo; not used by M2 extraction
└── VAE_latent_extraction_initial_setup.ipynb
```

## Milestone 1 Baseline

The Milestone 1 implementation is preserved under `milestone1/`. It validates
the project’s SML framing using Spotify/Kaggle acoustic features, PCA, and
K-Means. It is useful for narrative comparison, but M2 code should not depend on
its old root-level `src/` paths.

## Milestone 2 System

The M2 pipeline is split into two runtimes:

1. Colab/GPU extraction streams MTG-Jamendo samples, loads only the ACE-Step VAE
   via `diffusers`, extracts 64-D embeddings, and writes `artifacts/`.
2. Streamlit/CPU deployment loads finished FAISS, SQLite, metrics, and preview
   artifacts. It should never load the full ACE-Step LM/DiT inference stack.

Canonical project reference: `documentations/m2_implementation_handoff.md`.

Current merged artifact bundle:

- `1,734` unique tracks after deduplicating `2,000` extracted rows.
- `10` metadata-derived proxy playlist buckets.
- `431` artist ID rows and `684` album ID rows with display fallbacks.
- `30` generated synthetic playlists for multi-seed completion testing.

Expected generated artifacts live under `artifacts/merged/`:

- `vectors.index`
- `metadata.db`
- `embeddings.npy`
- `metrics.json`
- `manifest.json`
- `runtime_profile_*.png`
- `previews/*.mp3`

## Local Dashboard

Create a project-local virtual environment, install dependencies, and run
Streamlit:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The app can render a small demo fallback before final artifacts exist.

## Colab Extraction

Use `notebooks/one_shot_colab_extraction.ipynb` as the starting point for the
full MTG-Jamendo run. It is derived from
`VAE_latent_extraction_initial_setup.ipynb` but keeps the final extraction path
minimal and VAE-only.

## Evaluation And Deployment

After the Colab artifact shards are available locally, run the local pipeline
through the virtual environment:

```bash
.venv/bin/python scripts/merge_artifact_shards.py \
  artifacts/latent-recommend-artifacts-run1 \
  artifacts/latent-recommend-artifacts-run2 \
  --output artifacts/merged \
  --copy-previews

.venv/bin/python scripts/rebuild_metadata_model.py --artifacts artifacts/merged
.venv/bin/python scripts/validate_artifacts.py --artifacts artifacts/merged
.venv/bin/python scripts/evaluate_artifacts.py --artifacts artifacts/merged --k 10
.venv/bin/python scripts/plot_runtime_profile.py \
  artifacts/latent-recommend-artifacts-run1/runtime_profile.jsonl \
  artifacts/latent-recommend-artifacts-run2/runtime_profile.jsonl \
  --output-dir artifacts/merged

.venv/bin/streamlit run app.py
```

Deployment notes live in `documentations/deployment_strategy.md`.

For faster extraction, run two Colab jobs with five proxy buckets each and merge
the resulting artifact shards:

```bash
.venv/bin/python scripts/merge_artifact_shards.py \
  artifacts_shard_a artifacts_shard_b \
  --output artifacts --copy-previews
```
Recommended two-session split:

```python
%env TARGET_TAGS=ambient_soundscape,electronic_dance,classical_orchestral,jazz_soul,blues_roots
%env PER_TAG_LIMIT=200
```

and:

```python
%env TARGET_TAGS=hiphop_rap,folk_acoustic,indie_rock,experimental_trip_hop,happy_pop
%env PER_TAG_LIMIT=200
```

Runtime profiling logs from Colab can be plotted with:

```bash
.venv/bin/python scripts/plot_runtime_profile.py artifacts_*/runtime_profile.jsonl --output-dir artifacts
```

## Current Headline Metrics

From the current `artifacts/merged` bundle:

- Raw64 Precision@10: `0.2696`
- Raw64 Precision@4: `0.3098`
- MRR@10: `0.5205`
- Playlist completion HitRate@4: `0.6667`
- Playlist completion Precision@4: `0.2083`
- Triplet success: `0.597`
- Raw64 retrieval beats PCA10, PCA3, and Mahalanobis in the current ablation.
