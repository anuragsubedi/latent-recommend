# `latent-recommend`: Music Recommendation and Similarity Search via Learned Audio Embeddings

## 1. Abstract & Goal
Modern music recommendation systems are driven by collaborative filtering (what other users listen to). This inadvertently builds echo chambers of popularity bias, trapping obscure/indie artists at the bottom of the algorithm while hyper-promoting mainstream tracks.

**`latent-recommend`** fundamentally bypasses user-behavior metrics. It is a purely content-based search engine that evaluates the raw acoustic topology of music. By parsing audio through the ACE-Step 1.5 Diffusion Transformer VAE bottleneck, we extract deep latent embeddings. Applying Statistical Machine Learning (SML) techniques like PCA, K-Means clustering, and Nearest-Neighbors mapping allows us to recommend music *strictly by how it sounds*.

## 2. Milestone 1 (Baseline Release)
For Milestone 1, we establish the foundational metadata architecture and prove the core statistical theory using baseline surrogate metrics (standard acoustic features mapping) before integrating the heavy generative deep learning models in Milestone 2.

### 2.1 The Baseline Component
Our initial model ingests metadata and surrogate acoustic features (energy, valence, timbre/MFCC proxies) to formulate a statistical distribution test showcasing popularity bias. We then apply Principal Component Analysis (PCA) to demonstrate that tracks cluster acoustically independent of their mainstream popularity indexing.

## 3. Repository Structure (Milestone 1)
```text
latent-recommend/
├── README.md                           # Project overview
├── design_notes.md                     # Working memory & API logs
├── documentations/
│   ├── initial_roadmap.md              # Original step-by-step vision
│   ├── baseline_concepts.md            # Explanation of SML, PCA, & MFCCs
│   ├── milestone1_presentation.md      # Milestone 1 video presentation script & slides
│   └── ace_step/                       # ACE-Step base documentations
├── src/                                # Core Executables
│   ├── data_ingestion.py               # API ingestion to build the SQLite database
│   └── baseline_sml.py                 # Clustering, PCA, and Distribution rendering 
└── data/                               # Local data dir (generated on runtime)
    └── metadata.db                     # SQLite database tracking the tracks/features
```

## 4. Steps to Run & Reproduce (Baseline)
*Note: A Python 3.10+ environment is required.*

1. **Install Dependencies:**
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn sqlite3 requests librosa
   ```
2. **Ingest Data:**
   Run the data script to fetch acoustic profiles and generate the local SQLite database. 
   ```bash
   python src/data_ingestion.py
   ```
3. **Run Statistical Machine Learning Output:**
   Run the baseline modeling script to compute PCA, K-Means clusters, and render the graphs proving the bias hypothesis.
   ```bash
   python src/baseline_sml.py
   ```

## 5. References & Data Sources
- **Data APIs:** MusicBrainz, Last.fm, Spotify Web API (Proxy metrics).
- **Core ML Logic:** `scikit-learn` (PCA, Clustering), `librosa` (Audio processing concepts).
- **Generative Bottleneck:** [ACE-Step 1.5](https://huggingface.co/ACE-Step/Ace-Step1.5) open-source audio foundation model.
