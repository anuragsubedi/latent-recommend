import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'metadata_kaggle.db')
CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'pruned_spotify_tracks.csv')

def setup_database():
    """Initializes the expanded SQLite schema for the Kaggle dataset."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # 1. Core Track Metadata (Dropped explicit 'genre' as we are clustering unsupervised)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracks (
            track_id TEXT PRIMARY KEY,
            title TEXT,
            album TEXT,
            artist TEXT,
            year INTEGER,
            explicit BOOLEAN,
            duration_ms INTEGER
        )
    ''')
    
    # 2. Expanded Acoustic Features Table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS acoustic_features (
            track_id TEXT PRIMARY KEY,
            acousticness REAL,
            danceability REAL,
            energy REAL,
            instrumentalness REAL,
            liveness REAL,
            loudness REAL,
            speechiness REAL,
            valence REAL,
            tempo REAL,
            key INTEGER,
            mode INTEGER,
            time_signature REAL,
            FOREIGN KEY (track_id) REFERENCES tracks (track_id)
        )
    ''')
    conn.commit()
    return conn

def ingest_kaggle_data(conn):
    """Parses the Kaggle CSV and populates the SQLite tables."""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tracks")
    if cursor.fetchone()[0] > 0:
        print("Kaggle database already seeded.")
        return

    print(f"Reading CSV from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: {CSV_PATH} not found. Please run the pruning script first.")
        return
        
    df = pd.read_csv(CSV_PATH)
    
    tracks_data = []
    features_data = []
    
    for _, row in df.iterrows():
        # Parsing artist list (takes first artist safely or joins them)
        try:
            # Kaggle artists column is likely a string representation of a list: "['Artist 1', 'Artist 2']"
            artist_raw = row.get('artists', "['Unknown']")
            artist_clean = artist_raw.strip("[]").replace("'", "").split(',')[0].strip()
        except:
            artist_clean = "Unknown"
        
        # Track Tuple
        tracks_data.append((
            str(row['id']),
            str(row['name']),
            str(row['album']),
            artist_clean,
            int(row.get('year', 0)),
            bool(row.get('explicit', False)),
            int(row.get('duration_ms', 0))
        ))
        
        # Expanded Features Tuple
        features_data.append((
            str(row['id']),
            float(row.get('acousticness', 0.0)),
            float(row.get('danceability', 0.0)),
            float(row.get('energy', 0.0)),
            float(row.get('instrumentalness', 0.0)),
            float(row.get('liveness', 0.0)),
            float(row.get('loudness', 0.0)),
            float(row.get('speechiness', 0.0)),
            float(row.get('valence', 0.0)),
            float(row.get('tempo', 0.0)),
            int(row.get('key', -1)),
            int(row.get('mode', -1)),
            float(row.get('time_signature', 0.0))
        ))
        
    cursor.executemany("INSERT INTO tracks VALUES (?, ?, ?, ?, ?, ?, ?)", tracks_data)
    cursor.executemany("INSERT INTO acoustic_features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", features_data)
    conn.commit()
    print(f"Seeding complete. {len(tracks_data)} Spotify tracks ingested into {DB_PATH}.")

if __name__ == "__main__":
    connection = setup_database()
    ingest_kaggle_data(connection)
    connection.close()
