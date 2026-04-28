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

Expected generated artifacts:

- `artifacts/vectors.index`
- `artifacts/metadata.db`
- `artifacts/embeddings.npy`
- `artifacts/metrics.json`
- `artifacts/previews/*.mp3`

## Local Dashboard

Install the lightweight runtime dependencies and run Streamlit:

```bash
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

After the Colab artifact bundle is available locally:

```bash
python scripts/evaluate_artifacts.py
python scripts/validate_artifacts.py
streamlit run app.py
```

Deployment notes live in `documentations/m2/deployment_strategy.md`.

For faster extraction, run two Colab jobs with three tags each and merge the
resulting artifact shards:

```bash
python scripts/merge_artifact_shards.py \
  artifacts_ambient_dub_electronic artifacts_metal_classical_folk \
  --output artifacts --copy-previews
```

Runtime profiling logs from Colab can be plotted with:

```bash
python scripts/plot_runtime_profile.py artifacts_*/runtime_profile.jsonl --output-dir artifacts
```
