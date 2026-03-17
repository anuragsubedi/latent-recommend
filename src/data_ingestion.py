import sqlite3
import pandas as pd
import os
import requests
import random
import time

# Milestone 1 Rapid-Prototyping Data Setup
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'metadata.db')

def setup_database():
    """Initializes the SQLite schema for our baseline analysis."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create Track and Feature schema
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS tracks (
            track_id TEXT PRIMARY KEY,
            title TEXT,
            artist TEXT,
            popularity INTEGER,
            genre TEXT
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS acoustic_features (
            track_id TEXT PRIMARY KEY,
            acousticness REAL,
            danceability REAL,
            energy REAL,
            valence REAL,
            tempo REAL,
            FOREIGN KEY (track_id) REFERENCES tracks (track_id)
        )
    ''')
    conn.commit()
    return conn

def seed_dummy_baseline_data(conn):
    """
    For Milestone 1 Rapid Prototyping, we simulate API ingestion to demonstrate the statistical proofs.
    In M2, this will be replaced with heavy Spotify/MusicBrainz network queries.
    We are generating 200 synthetic tracks spanning mainstream and obscure popularity, 
    but clustering them into 3 distinct acoustic profiles to prove our hypothesis.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM tracks")
    if cursor.fetchone()[0] > 0:
        print("Database already seeded.")
        return

    print("Seeding SQLite with synthetic baseline tracks representing different popularity and acoustic profiles...")
    
    genres = ["Ambient", "Dub-Techno", "Pop-Mainstream"]
    tracks = []
    features = []

    for i in range(200):
        track_id = f"TRK_{i:04d}"
        genre = random.choice(genres)
        
        # Simulate popularity bias: Pop is mostly highly popular (80-100), Ambient/Dub are obscure (0-30)
        # Note: We sprinkle a few obscure pops and popular ambients to test distribution overlapping.
        if genre == "Pop-Mainstream":
            popularity = int(random.gauss(85, 10))
        else:
            popularity = int(random.gauss(20, 15))
        popularity = max(0, min(100, popularity))
        
        tracks.append((track_id, f"Song {i}", f"Artist {i}", popularity, genre))
        
        # Simulate surrogate acoustic features (acting like extracted MFCCs/Latents)
        if genre == "Ambient":
            ac, da, en, va, te = (random.gauss(0.9, 0.1), random.gauss(0.2, 0.1), random.gauss(0.1, 0.1), random.gauss(0.2, 0.1), random.gauss(80, 10))
        elif genre == "Dub-Techno":
            ac, da, en, va, te = (random.gauss(0.1, 0.1), random.gauss(0.8, 0.1), random.gauss(0.6, 0.1), random.gauss(0.4, 0.1), random.gauss(120, 5))
        else: # Pop-Mainstream
            ac, da, en, va, te = (random.gauss(0.2, 0.1), random.gauss(0.7, 0.1), random.gauss(0.8, 0.1), random.gauss(0.8, 0.1), random.gauss(110, 15))
            
        features.append((track_id, max(0, min(1, ac)), max(0, min(1, da)), max(0, min(1, en)), max(0, min(1, va)), te))
        
    cursor.executemany("INSERT INTO tracks VALUES (?, ?, ?, ?, ?)", tracks)
    cursor.executemany("INSERT INTO acoustic_features VALUES (?, ?, ?, ?, ?, ?)", features)
    conn.commit()
    print("Seeding complete. 200 tracks ingested.")

if __name__ == "__main__":
    connection = setup_database()
    seed_dummy_baseline_data(connection)
    connection.close()
