"""
ISRO/STScI TESS Exoplanet CTL Ingestion Pipeline
Parses the Exoplanet Candidate Target List (xCTL v08.01) CSV from STScI.

Usage:
    1. Download xCTL from https://archive.stsci.edu/tess/tic_ctl.html
    2. Place the CSV at data/xctl_v0801.csv
    3. Run: python ingest_isro_ctl.py
"""
import pandas as pd
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
OUTPUT_PATH = os.path.join(DATA_DIR, "processed_ctl_targets.csv")

def process_ctl(csv_path=None):
    """
    Ingests the STScI xCTL CSV and filters for high-priority exoplanet targets.

    Filtering criteria (physics bounds for optimal transit detectability):
    - Dwarf stars only: radius < 2.0 solar radii
    - Temperature bounds: 3000K < Teff < 7000K
    - Brightness: Tmag < 13.0 (sufficient photon count for high SNR)
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "xctl_v0801.csv")

    if not os.path.exists(csv_path):
        print(f"ERROR: CTL CSV not found at {csv_path}")
        print("Please download it from: https://archive.stsci.edu/tess/tic_ctl.html")
        print("  -> Download The Exoplanet CTL (the 'xCTL', v08.01): (.csv) (497 MB)")
        sys.exit(1)

    print(f"Reading CTL from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total entries: {len(df)}")

    required_cols = ["ID", "Tmag", "Teff", "mass", "rad"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing expected columns: {missing}")
        print(f"Available columns: {list(df.columns[:20])}...")
        return

    df_filtered = df[
        (df["rad"] < 2.0) &
        (df["Teff"] > 3000) &
        (df["Teff"] < 7000) &
        (df["Tmag"] < 13.0)
    ].copy()

    df_filtered = df_filtered.dropna(subset=required_cols)
    print(f"After filtering: {len(df_filtered)} targets")

    df_out = df_filtered[required_cols].head(500)

    os.makedirs(DATA_DIR, exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df_out)} processed targets to {OUTPUT_PATH}")

if __name__ == "__main__":
    process_ctl()
