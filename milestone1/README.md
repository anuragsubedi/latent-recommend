# `latent-recommend`: Music Search via Learned Audio Embeddings

## 1. Abstract & Goal

Modern music recommendation systems are driven by collaborative filtering (what other users listen to). This inadvertently builds echo chambers of popularity bias, trapping obscure/indie artists at the bottom of the algorithm while hyper-promoting mainstream tracks.

**`latent-recommend`** fundamentally bypasses user-behavior metrics. It is a purely content-based search engine that evaluates the raw acoustic topology of music. By parsing audio through the ACE-Step 1.5 Diffusion Transformer VAE bottleneck, we extract deep latent embeddings. Applying Statistical Machine Learning (SML) techniques like PCA, K-Means clustering, and Nearest-Neighbors mapping allows us to recommend music *strictly by how it sounds*.

## 2. Milestone 1 (Baseline Release)

For Milestone 1, we establish the foundational metadata architecture and prove the core statistical theory using a massive **1.2M Spotify Kaggle Dataset**. By using pre-extracted proxy acoustic features (energy, valence, timbre mapping), we mathematically validate the clustering topologies required before stepping into heavy deep learning infrastructure in Milestone 2.

### 2.1 The Baseline SML Components

1. **The Kaggle Migration**: We utilize the `Rodolfofigueroa/spotify-12m-songs` corpus, pruning it to 1,000 highly curated tracks to populate an explicit 13-feature array inside SQLite.
2. **Feature Pruning & Clustering**: We drop non-acoustic rigid identifiers (Key, Time Signature) to create an **8D Array** of raw acoustic feeling. We process this array through a **Unsupervised K-Means clustering algorithm** which natively discovers distinct geometric neighborhoods that we call "Organic Genres".
3. **Dimensionality Reduction**: We apply Principal Component Analysis (PCA) to squash the 8 dimensions down into a **3D Latent Space Map**, proving that obscure tracks natively map directly adjacent to mainstream hits strictly due to their acoustic waveform geometry.

## 3. Repository Structure (Milestone 1 Core)

```text
latent-recommend/
├── README.md                           # Project overview
├── design_notes.md                     # Architectural log of API bottlenecks and M2 pivots
├── documentations/
│   ├── baseline_concepts.md            # Foundations of SML, PCA, & M1 vs M2 pipelines
├── src/                            
│   ├── scripts/prune_spotify_data.py   # Kagglehub fetcher to isolate the 1,000 track baseline
│   ├── data_ingestion_kaggle.py        # SQLite builder mapping the massive 13-feature array
│   ├── baseline_sml_kaggle.py          # 8D-to-3D PCA dimensionality reduction and clustering
│   ├── data_ingestion.py               # Legacy synthetic test
│   └── baseline_sml.py                 # Legacy hypothesis analysis
└── data/                               # Local data dir (generated on runtime)
    ├── pruned_spotify_tracks.csv       # The isolated dataset
    └── metadata_kaggle.db              # Active SQLite database
```

## 4. Steps to Run & Reproduce

*Note: A Python 3.10+ virtual environment is required.*

1. **Install Dependencies:**
   Install required packages via `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
2. **Fetch and Prune the Kaggle CSV:**
   This script securely downloads the massive Kaggle dataset and isolates your 1000-track subset.

   ```bash
   python src/scripts/prune_spotify_data.py
   ```
3. **Build the Audio Database:**
   Run the ingestion script to parse the CSV and build the local SQLite environment.

   ```bash
   python src/data_ingestion_kaggle.py
   ```
4. **Execute Statistical Machine Learning Matrix:**
   Run the baseline modeling script to compute the K-Means clusters and render the 3D PCA scatterplots.

   ```bash
   python src/baseline_sml_kaggle.py
   ```

## 5. References & Data Sources

- **Data APIs**: Spotify 1.2M Songs via Kaggle (`Rodolfofigueroa/spotify-12m-songs`)
- **Core ML Logic**: `scikit-learn` (PCA, Unsupervised Clustering Matrix)
- **Generative Bottleneck**: [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) open-source audio foundation model.
