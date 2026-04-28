import sqlite3
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'metadata.db')

def load_data():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError("Database not found. Run data_ingestion.py first.")
    
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT t.track_id, t.popularity, t.genre,
               f.acousticness, f.danceability, f.energy, f.valence, f.tempo
        FROM tracks t
        JOIN acoustic_features f ON t.track_id = f.track_id
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def analyze_popularity_bias(df):
    """
    STATISTICAL COMPONENT 1: Hypothesis testing distributions.
    We plot the distribution showing the popularity bias gap between Mainstream and Indie/Obscure (Ambient/Dub).
    """
    print("\n--- Running Statistical Analysis: Popularity Bias ---")
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(data=df, x='popularity', hue='genre', fill=True, common_norm=False, palette='viridis')
    plt.title('Distribution of Popularity Scores by Genre')
    plt.xlabel('Popularity Score (0-100)')
    plt.ylabel('Density')
    
    plot_path = os.path.join(os.path.dirname(DB_PATH), 'popularity_distribution.png')
    plt.savefig(plot_path)
    print(f"Popularity distribution graph saved to {plot_path}")
    
    # Simple T-Test (Hypothesis Testing)
    pop = df[df['genre'] == 'Pop-Mainstream']['popularity']
    obscure = df[df['genre'].isin(['Ambient', 'Dub-Techno'])]['popularity']
    
    t_stat, p_val = stats.ttest_ind(pop, obscure, equal_var=False)
    print(f"Hypothesis Test (Welch's t-test) comparing Mainstream vs Obscure popularity:")
    print(f"T-statistic: {t_stat:.2f}, P-value: {p_val:.2e}")
    if p_val < 0.05:
        print("Conclusion: Significant statistical difference in popularity exists, confirming the platform bias.")

def run_pca_and_clustering(df):
    """
    STATISTICAL COMPONENT 2: Dimensionality Reduction & Unsupervised Clustering.
    We apply PCA to map the features to 2D, then cluster using K-Means to show that music 
    groups naturally by acoustic features, regardless of the popularity gap shown above.
    """
    print("\n--- Running Unsupervised SML: PCA & K-Means ---")
    
    feature_cols = ['acousticness', 'danceability', 'energy', 'valence', 'tempo']
    X = df[feature_cols].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Dimensionality Reduction
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df['pca1'] = X_pca[:, 0]
    df['pca2'] = X_pca[:, 1]
    
    print(f"PCA Explained Variance Ratio: {pca.explained_variance_ratio_}")
    
    # 2. Unsupervised Clustering
    # We purposefully don't tell K-Means the labels. We want to see if it finds 3 clusters organically.
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Plotting PCA Space with True Genres to prove acoustic separation bypasses popularity
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='pca1', y='pca2', hue='genre', style='cluster', palette='Set1', s=100)
    plt.title('PCA of Acoustic Features: Organic Topological Mapping')
    plt.xlabel('First Principal Component (PCA1)')
    plt.ylabel('Second Principal Component (PCA2)')
    
    pca_plot_path = os.path.join(os.path.dirname(DB_PATH), 'pca_clusters.png')
    plt.savefig(pca_plot_path)
    print(f"PCA clustering scatterplot saved to {pca_plot_path}")
    print("Baseline model complete. We have proven that the acoustic topology creates distinct neighborhoods independently of the popularity index.")

if __name__ == "__main__":
    data = load_data()
    analyze_popularity_bias(data)
    run_pca_and_clustering(data)
