# **Implementation Roadmap: Project `latent-recommend`**

## **1\. Project Overview and Objectives**

The objective of the `latent-recommend` project is to build a purely content-based music recommendation and similarity search engine. Modern recommendation systems predominantly rely on collaborative filtering (user-behavior metrics), which creates a "popularity bias" echo chamber.

This project bypasses collaborative filtering entirely. Instead, we will rely on the acoustic properties of the music. By extracting compact latent vectors from a structurally diverse music corpus, we will apply classical Statistical Machine Learning (SML) techniques—specifically dimensionality reduction, unsupervised clustering, and nearest-neighbor retrieval—to map the natural topology of musical genres and enable purely content-driven similarity search.

## **2\. Parallel Ecosystem Context: Project `neural-noise`**

*Contextual note for development agents: This section outlines a parallel project. The agents assigned to `latent-recommend` are not responsible for building `neural-noise`, but must understand the shared architectural dependencies.*

In parallel to this SML pipeline, a Deep and Generative AI (DGAI) project titled `neural-noise` is being developed. `neural-noise` focuses on mechanistic interpretability and controllable music generation via latent space navigation within Diffusion Transformers (specifically the open-source ACE-Step 1.5 model). The goal of `neural-noise` is to achieve "style steering" (e.g., guiding generation toward ambient/dub-techno characteristics) by intervening in the reverse diffusion process or the VAE latent bottleneck.

**The Architectural Link:** To maintain a unified infrastructure, `latent-recommend` will utilize the exact same VAE/Encoder from the ACE-Step 1.5 model to compress audio snippets into latent vectors. The embeddings stored in the `latent-recommend` database might be foundational map for the `neural-noise` project.

## **3\. Step-by-Step Implementation Plan**

Here's a rough step-by-step plan I had formulated earlier for the latent-recommend project. 

### **Step 1: The Metadata Skeleton & Bias Mitigation**

Before processing audio, a robust, queryable metadata backbone must be established. The critical requirement of this phase is mitigating popularity bias to ensure obscure, independent tracks are represented equally alongside mainstream music.

* **Database Schema Design:** \* Initialize a relational database (SQLite for the preliminary phase, scaling to PostgreSQL with `pgvector` for embedding storage).
  * Required entities: `Track`, `Artist`, `Album`, `GenreTags`, and a dedicated `PopularityIndex` metric.
* **Data Ingestion & Stratified Sampling:**
  * Utilize APIs such as MusicBrainz (optimal for deep metadata and obscure tracks) and Last.fm (for genre tagging and popularity heuristics).
  * Implement a stratified sampling algorithm. The ingestion script must actively query for equal distributions across popularity percentiles (e.g., 20% mainstream, 40% mid-tier, 40% obscure/indie).
  * Ensure heavy sampling of specific boundary-condition genres of interest, specifically ambient soundscapes, dub-techno, and classical pieces.

### **Step 2: Audio Acquisition & Shared Latent Extraction**

Once the metadata skeleton is populated, the pipeline must acquire the raw acoustic data and process it through the shared generative bottleneck.

**(IMPORTANT!! ) A preliminary requirement before starting this phase will be to get the code for the Ace-Step 1.5 model up and working and explore all the available models. We should also dive deep into the DiT VAE encoder of the AceStep model and how we can capture and make sense of the latent vectors.**

* **Audio Fetching & Standardization:**
  * Programmatically fetch 30-second audio previews (via Spotify API, 7digital, or open-source datasets matching our database IDs).
  * Standardize the audio format: uniform sample rate (e.g., 44.1kHz), converted to mono, and trimmed to exact lengths.
* **Latent Embedding Generation:**
  * Pass the standardized audio arrays through the pre-trained ACE-Step 1.5 VAE encoder.
  * Extract the dense latent vectors and store them as arrays directly in the database, linked via Foreign Key to the `Track` metadata records.

### **Step 3: Statistical Learning & Topology Mapping**

With the database populated with both metadata and latent vectors, classical SML techniques will be applied to analyze the acoustic manifold.

* **Dimensionality Reduction:**
  * Apply Principal Component Analysis (PCA) and t-Distributed Stochastic Neighbor Embedding (t-SNE) to the latent vectors.
  * Generate 2D/3D visual mappings to evaluate if the model naturally separates distinct acoustic profiles (e.g., a dense dub-techno track vs. a sparse acoustic track) purely based on the embeddings, without metadata labels.
* **Unsupervised Clustering:**
  * Implement K-Means and Hierarchical clustering algorithms on the high-dimensional embeddings.
  * Evaluate the resulting clusters: Determine if they map to traditional metadata genres or if they discover new, purely acoustic categorizations (e.g., clustering by tempo-synced basslines or specific harmonic structures).

### **Step 4: Retrieval Engine Implementation and WebApp deployment**

The final phase of the SML project is building the actual recommendation interface based on the generated clusters and latent geometry.

* **Similarity Search:**
  * Implement K-Nearest Neighbors (K-NN) and Cosine Similarity metrics.
  * Build a query function where a target `Track ID` is provided, and the engine retrieves the top $N$ closest vectors in the latent space.
* **Performance Evaluation:**
  * Compare the retrieved "closest" tracks against baseline collaborative filtering outputs to quantify the differences in recommendation diversity and bias mitigation.
