import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'metadata_kaggle.db')

def load_data():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Database not found. Run data_ingestion_kaggle.py first.")
    
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT t.track_id, t.title, t.artist,
               f.acousticness, f.danceability, f.energy, f.instrumentalness, 
               f.liveness, f.loudness, f.speechiness, f.valence, f.tempo,
               f.key, f.mode, f.time_signature
        FROM tracks t
        JOIN acoustic_features f ON t.track_id = f.track_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def run_unsupervised_pipeline(df):
    """
    Executes PCA and K-Means on the expanded feature set without any explicit genre supervision.
    We will map the 13 acoustic features down to 2 Principal Components.
    """
    print("\n--- Running Kaggle SML Pipeline: Expanded PCA & K-Means ---")
    
    # 1. Feature Selection (Pruned to 9 Core Acoustic Dimensions)
    # Dropped 'key', 'mode', 'time_signature', and 'loudness' as they introduce non-acoustic/structural noise
    # into the K-Means clustering, preventing organic topological mapping.
    feature_cols = [
        'acousticness', 'danceability', 'energy', 'instrumentalness', 
        'liveness', 'speechiness', 'valence', 'tempo'
    ]
    X = df[feature_cols].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Dimensionality Reduction (8D -> 3D)
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    df['pca3'] = X_pca[:, 2]
    
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    
    # 3. Unsupervised Clustering
    # We ask K-Means to find 4 organic clusters in the 8-dimensional space
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 4. Organic Genre Assignment Analysis (Evaluate prior to plotting to annotate the legend)
    print("\n--- Organic Cluster Analysis ---")
    cluster_means = df.groupby('cluster')[feature_cols].median()
    cluster_labels = {}
    
    for cluster_id in range(n_clusters):
        stats = cluster_means.loc[cluster_id]
        
        # Simple heuristic organic tagging based on Audio Features
        if stats['acousticness'] > 0.6 and stats['energy'] < 0.4:
            organic_tag = "Acoustic / Ambient / Chill"
        elif stats['energy'] > 0.7 and stats['danceability'] > 0.6:
            organic_tag = "High-Energy / Club / Pop"
        elif stats['instrumentalness'] > 0.5 and stats['danceability'] < 0.5:
            organic_tag = "Atmospheric / Classical"
        else:
            organic_tag = "Mid-Tempo / Standard Band"
            
        cluster_labels[cluster_id] = f"{organic_tag}"
        print(f"\n[Cluster {cluster_id}] Profile: {organic_tag}")
        print(f"  - Energy/Dance: {stats['energy']:.2f} / {stats['danceability']:.2f}")
        print(f"  - Acoustic/Instrumental: {stats['acousticness']:.2f} / {stats['instrumentalness']:.2f}")
        print(f"  - Valence (Positivity): {stats['valence']:.2f}")

    df['organic_genre'] = df['cluster'].map(cluster_labels)

    # 5. Plotting (3D Projection)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color palette for clusters
    colors = ['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4']
    
    for cluster_id in range(n_clusters):
        subset = df[df['cluster'] == cluster_id]
        ax.scatter(
            subset['pca1'], subset['pca2'], subset['pca3'], 
            c=colors[cluster_id % len(colors)], 
            label=cluster_labels[cluster_id], 
            s=40, alpha=0.7
        )
        
    ax.set_title('Unsupervised Kaggle Pipeline: 8D -> 3D Acoustic PCA Mapping')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend(title="Organic Genre Map", loc='best')

    pca_plot_path = os.path.join(os.path.dirname(DB_PATH), 'kaggle_pca_clusters_3d.png')
    plt.savefig(pca_plot_path)
    print(f"\nKaggle 3D PCA clustering scatterplot saved to {pca_plot_path}")

if __name__ == "__main__":
    data = load_data()
    run_unsupervised_pipeline(data)
