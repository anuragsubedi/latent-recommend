# Baseline Concepts & SML Fundamentals

This document tracks the core mathematical and statistical concepts utilized across the `latent-recommend` ecosystem. It explicitly delineates the **Milestone 1 Baseline Model** (which uses proxy metrics) from the **Milestone 2 Model** (which will use true generative latent vectors).

## 1. What are Audio Features? 

Before utilizing complex VAE outputs from deep generative models, we need quantifiable representations of sound. 

### 1.1 The "Gold Standard" (MFCCs & True Acoustic Parsing)
- **Concept**: MFCCs (Mel-Frequency Cepstral Coefficients) are a mathematically compact representation of the short-term power spectrum of a sound. They represent the "timbre" or "shape" of the audio signal exactly as the human ear perceives it.
- **Why it matters**: Human hearing is non-linear. The **Mel-Scale** maps frequencies to match human perception. When extracted (via libraries like `librosa`), this yields high-dimensional arrays summarizing the track's sonic profile (e.g., vocal warmth, heavy bass). 
- ***Note on M1 vs M2***: Extracting true MFCC arrays or raw audio vectors from tens of thousands of songs computationally expensive. This exact raw audio parsing is reserved for the DiT VAE bottleneck in Milestone 2.

### 1.2 The Milestone 1 "Proxy" (Spotify Audio Features)
For the Milestone 1 rapid-prototyping phase, we **do not** extract actual MFCCs from raw audio. Instead, we use heuristic *proxies* (simulating data we would fetch from a platform like Spotify's API) to represent the acoustic topology:
- **Acousticness**: Confidence measure of whether the track is acoustic vs. electronic.
- **Energy / Loudness**: Perceptual measure of intensity and activity.
- **Valence**: Musical positiveness conveyed by a track.
- **Danceability & Tempo**: Structural rhythmic markers.

By structuring these 5 variables into an array, we simulate the high-dimensional vectors we will eventually handle in M2, allowing us to build out the statistical pipeline immediately.

---

## 2. Solving the "Popularity Bias" Echo Chamber

The entire premise of `latent-recommend` revolves around the **Popularity Bias Problem** inherent in modern music streaming.

### 2.1 The Collaborative Filtering Trap
Current recommendation engines (Spotify, Apple Music) primarily use *Collaborative Filtering*. They do not analyze the music itself; they analyze user behavior arrays. 
- If User A and User B listen to the same mainstream Pop track, the algorithm links them.
- If User A listens to an obscure Indie track, the algorithm recommends it to User B.
- **The Problem**: An artist with zero initial listeners has zero connections in the matrix. Their music, regardless of how objectively incredible it sounds, is NEVER pushed to new listeners. The algorithm inherently compounds mainstream popularity, burying underground music.

### 2.2 Our Solution & Hypothesis Testing
We aim to recommend music strictly by **how it sounds** (Acoustic Content-Based Filtering), ignoring the behavioral matrix entirely. 
- **The M1 Statistical Proof**: To justify this project, we must prove the bias exists. In `baseline_sml.py`, we map the distributions of mainstream tracks vs obscure tracks on a popularity axis.
- **Hypothesis Testing**: We use statistical testing (like Welch's T-Test) to prove that the popularity index of these groups is not equal. If we can subsequently show (via clustering) that these unequal groups actually share the exact same acoustic space, we prove that our engine bypasses the bias filter.

---

## 3. Dimensionality Reduction Pipeline (M1 -> M2)

Audio feature extraction produces highly dimensional arrays. 

### 3.1 The Process (PCA)
- **The Problem**: Visualizing relationships in high dimensions is impossible. Calculating distances in massive dense vectors (like the ones from ACE-Step) suffers from the "curse of dimensionality", where distance metrics become meaningless.
- **Principal Component Analysis (PCA)**: A statistical technique that orthogonalizes dimensions. It finds the "directions" (Principal Components) along which the variance of the data is maximized, compressing the data into a mathematically dense representation.

### 3.2 Milestone 1 Prototype (The "Toy" Pipeline)
In our current `baseline_sml.py`, we are doing a highly simplified version of this pipeline:
- We start with only **5 dimensions** (our acoustic proxies: Energy, Valence, Danceability, Acousticness, Tempo).
- We use PCA to shrink these 5 dimensions down to **2 dimensions** (PCA1, PCA2).
- *Rationale*: While squashing 5 dimensions to 2 might seem trivial, it establishes the entire mathematical pipeline (scaling -> matrix transformation -> extraction of explained variance -> mapping coordinates).

### 3.3 Milestone 2 (The True Pipeline)
When we transition to Milestone 2, this exact pipeline must be ready to handle massive scale:
- The ACE-Step 1.5 VAE Encoder outputs **temporal latent arrays**. A 30-second audio clip might be represented by thousands of continuous float values (e.g., `tensor([1, 16, 86])` or specifically `[Channels, Latent_Dim, Sequence_Length]`).
- We will flatten or pool these arrays into dense multi-hundred-dimensional vectors.
- At that scale, standard PCA will be utilized heavily, along side advanced non-linear reducers like **t-SNE** or **UMAP**, to map the complex acoustic manifold into 3D clusters for our WebApp.

---

## 4. Unsupervised Clustering

Our core thesis relies on grouping music strictly via acoustic proximity to avoid collaborative filtering bias.

### K-Means Clustering
- **The Process**: K-Means groups our tracks into $K$ distinct clusters based on their coordinate distance in the newly compressed PCA space.
- **The Application**: We determine mathematically what constitutes a "neighborhood" organically by the soundwaves, rather than explicit user tagging. An obscure track and a mainstream hit might sit right next to each other in this space, acting as perfect counterparts for the final recommendation engine.
