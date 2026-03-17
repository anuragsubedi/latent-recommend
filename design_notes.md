# `latent-recommend` Design Notes & Architectural Log

## 1. Project Philosophy
- **Goal**: A purely content-based music recommendation engine using latent audio space geometry, bypassing collaborative filtering (popularity bias).
- **Core Technology**: Extracting latent features using the ACE-Step 1.5 VAE Encoder, followed by Statistical Machine Learning (SML) for clustering and similarity.

## 2. Milestone 1 (M1) Baseline Strategy
- **Baseline Model**: Before navigating the heavy VAE embeddings, we need to prove the statistical concept. The baseline utilizes a mock-ingestion script to generate "surrogate" acoustic features (simulating standard metrics like Energy, Valence, Acousticness).
- **Validation**: The pipeline executes dimensionality reduction (PCA) against these simple 5-feature arrays, allowing us to perform K-Means clustering and statistically prove that topological boundaries form strictly on sound, not popularity.

## 3. Data Ingestion & API Assumptions
- **Sources (Actual vs Proxy)**: 
  - *MusicBrainz*: Excellent for deep, structured metadata and obscure tracks, but rate-limited (typically 1 req/sec). 
  - *Spotify API*: Excellent for rapid fetching of baseline acoustic metrics.
- **Rate-Limiting Bottleneck (M1 constraints)**: Ingesting enough tracks from MusicBrainz at 1 req/sec to do meaningful SML is prohibitive for the initial deadline.
- **Triage**: As an alternative rapid-proxy, `data_ingestion.py` seeds our SQLite baseline with synthetic artists across popularity tiers mapped to explicit acoustic metrics, immediately populating the DB so our `baseline_sml.py` statistical proofs can fire. MusicBrainz pulling and raw audio processing are backlogged for the deeper M2 deliverables.

## 4. Latent Extraction Engine (Transition from M1 -> M2)
- **The Problem**: In M1, we squash 5 proxy dimensions to 2. This is a "toy" pipeline simply validating the math.
- **The M2 DiT Scale**: In M2, we will hit the ACE-Step `tiled_encode` bottleneck. It compresses raw audio into dense temporal convolutions (massive arrays). 
- **Bottlenecks**: Processing raw `.wav` files through DiT VAE will require heavy GPU availability or significant CPU time. We must ensure that our `.db` mappings can handle the size of these NumPy arrays before running PCA vectors on them. 
- **Storage**: We are currently using SQLite. In Phase 2, we will likely need to migrate to PostgreSQL with `pgvector` for efficient K-NN and cosine similarity searches across the vector density.

## 5. Shared Infrastructure with `neural-noise`
- The `latent-recommend` database acts as a foundational map for `neural-noise` latent space navigation.
- Both projects rely heavily on interpreting the exact same VAE outputs from the ACE-Step 1.5 codebase. The PCA geometries generated in `latent-recommend` might define the actual "Style Sliders" (e.g., sliding around the PCA grid to change a generative outcome).
