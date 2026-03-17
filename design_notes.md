# `latent-recommend` Design Notes & Architectural Log

## 1. Project Philosophy
- **Goal**: A purely content-based music recommendation engine using latent audio space geometry, bypassing collaborative filtering (popularity bias).
- **Core Technology**: Extracting latent features using the ACE-Step 1.5 VAE Encoder, followed by Statistical Machine Learning (SML) for clustering and similarity.

## 2. Milestone 1 (M1) Baseline Strategy
- **Baseline Model**: Before navigating the heavy VAE embeddings, we need to prove the statistical concept. The baseline utilizes a 1,000-track subset curated from a massive 1.2M Spotify Kaggle dataset, parsing standard acoustic metrics (Energy, Valence, Acousticness).
- **Validation**: The pipeline executes dimensionality reduction (PCA) against an 8-feature subset array, allowing us to perform Unsupervised K-Means clustering. This generates native "Organic Genres" entirely stripped of human behavioral bias, proving that topological boundaries form strictly on sound, not popularity.
- **Topology Refinement (Feature Pruning)**: Initially mapping 13 API dimensions yielded "messy" 2D topologies. Mathematical noise was introduced by rigid, non-perceptual identifiers like `key`, `mode`, and `time_signature`. Pruning out these structural metrics allowed algorithms to map raw *acoustic feeling*, yielding highly accurate clustering when projected into a **3D PCA Space**.

## 3. Data Ingestion & API Assumptions
- **Sources (Actual vs Proxy)**: 
  - *MusicBrainz*: Excellent for deep, structured metadata and obscure tracks, but severely rate-limited (typically 1 req/sec). 
  - *Spotify API (via Kaggle)*: We bypassed rate limiting entirely for M1 by ingesting the `Rodolfofigueroa/spotify-12m-songs` Kaggle dataset.
- **Rate-Limiting Bottleneck (M1 constraints)**: Ingesting enough tracks from MusicBrainz at 1 req/sec to do meaningful SML is prohibitive for the initial deadline.
- **Triage**: As a rapid baseline for M1, we curated `pruned_spotify_tracks.csv` to seed our SQLite database. This populates our exact 13-feature array requirements locally. MusicBrainz pulling and raw audio processing remain backlogged for the deeper M2 deliverables.

## 4. Latent Extraction Engine (Transition from M1 -> M2)
- **The Problem**: In M1, we squash 8 proxy dimensions to 3. This is a baseline pipeline simply validating the math on pre-extracted platform metrics.
- **The M2 DiT Scale**: In M2, we will hit the ACE-Step `tiled_encode` bottleneck. It compresses raw audio into dense temporal convolutions (massive arrays). 
- **Bottlenecks**: Processing raw `.wav` files through DiT VAE will require heavy GPU availability or significant CPU time. We must ensure that our `.db` mappings can handle the size of these NumPy arrays before running PCA vectors on them. 
- **Storage**: We are currently using SQLite. In Phase 2, we will likely need to migrate to PostgreSQL with `pgvector` for efficient K-NN and cosine similarity searches across the vector density.

## 5. Shared Infrastructure with `neural-noise`
- The `latent-recommend` database acts as a foundational map for `neural-noise` latent space navigation.
- Both projects rely heavily on interpreting the exact same VAE outputs from the ACE-Step 1.5 codebase. The PCA geometries generated in `latent-recommend` might define the actual "Style Sliders" (e.g., sliding around the PCA grid to change a generative outcome).
