import kagglehub
import pandas as pd
import os
import glob

def prune_dataset():
    print("Downloading/Locating dataset from Kaggle...")
    # This downloads the dataset to a local cache and returns the path
    path = kagglehub.dataset_download("rodolfofigueroa/spotify-12m-songs")
    print("Dataset directory:", path)
    
    # Locate the CSV file inside the downloaded directory
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    if not csv_files:
        print("Error: No CSV files found in the dataset directory!")
        return
        
    data_file = csv_files[0]
    print(f"Loading {data_file} into Pandas (~300MB)...")
    
    # Read the dataset
    df = pd.read_csv(data_file)
    print(f"Original dataset shape: {df.shape}")
    print("Available columns:", df.columns.tolist())
    
    # ---------------------------------------------------------
    # STRATEGIC PRUNING LOGIC
    # ---------------------------------------------------------
    # We want 1000 rows representing exactly 10 distinct genres.
    # Let's see if the dataset has a genre column.
    
    # Some datasets have it named differently.
    genre_col = None
    for col in ['genre', 'track_genre', 'genres']:
        if col in df.columns:
            genre_col = col
            break
            
    if not genre_col:
        print("\nWARNING: No clear 'genre' column found! Falling back to random sample.")
        # If there's no genre column, we just take a 1000-row random sample
        pruned_df = df.sample(n=1000, random_state=42)
    else:
        # If a genre column exists, let's identify the top 10 most common genres
        print(f"\nFound genre column: '{genre_col}'")
        
        # We might want to explicitly pick some distinct ones (like your baseline: Ambient, Techno, Pop)
        # Or just pick the 10 most frequent to guarantee we have enough data per genre.
        top_10_genres = df[genre_col].value_counts().nlargest(10).index.tolist()
        print(f"Selected Top 10 Genres: {top_10_genres}")
        
        # Filter the dataset to ONLY include these 10 genres
        df_filtered = df[df[genre_col].isin(top_10_genres)]
        
        # Sample exactly 100 tracks per genre to avoid class imbalance in our clustering
        print("Sampling 100 tracks per genre...")
        pruned_df = df_filtered.groupby(genre_col, group_keys=False).apply(
            lambda x: x.sample(n=100, random_state=42) if len(x) >= 100 else x
        )
        
    print(f"Pruned dataset shape: {pruned_df.shape}")
    
    # Prepare the output path in the `data/` directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(project_root, "data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_csv = os.path.join(output_dir, "pruned_spotify_tracks.csv")
    pruned_df.to_csv(output_csv, index=False)
    print(f"\n✅ Success! Pruned dataset saved to: {output_csv}")

if __name__ == "__main__":
    prune_dataset()
