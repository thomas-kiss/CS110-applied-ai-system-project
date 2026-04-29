"""
One-time script: clean and prepare the full Kaggle Spotify dataset.

Usage:
    python scripts/prepare_data.py --input data/kaggle_raw.csv

Expects the raw Kaggle CSV at data/kaggle_raw.csv (extract from the downloaded zip).
Writes data/kaggle_tracks.csv with columns matching the existing schema.
"""

import argparse
import pandas as pd


def prepare(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)

    # Rename columns to match existing schema
    df = df.rename(columns={
        "track_name":  "title",
        "artists":     "artist",
        "track_genre": "genre",
        "tempo":       "tempo_bpm",
    })

    # Keep only the columns score_song() needs
    keep = ["title", "artist", "genre", "danceability", "energy",
            "acousticness", "valence", "tempo_bpm"]
    df = df[keep].dropna()

    # Cast numeric columns
    for col in ["energy", "valence", "danceability", "acousticness", "tempo_bpm"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()

    # Drop exact duplicate title+artist pairs
    df = df.drop_duplicates(subset=["title", "artist"]).reset_index(drop=True)

    # Add sequential id
    df.insert(0, "id", range(1, len(df) + 1))

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} tracks to {output_path}")
    print(f"Genres represented: {df['genre'].nunique()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/kaggle_raw.csv")
    parser.add_argument("--output", default="data/kaggle_tracks.csv")
    args = parser.parse_args()
    prepare(args.input, args.output)