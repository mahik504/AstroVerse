import os
import argparse
import pandas as pd
import logging
from src.tic_fetcher import fetch_tic_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_metadata(labels_csv, output_csv):
    """
    Reads the TIC IDs from the labels CSV and fetches 
    stellar metadata (Mass, Radius, Teff, etc.) from the TIC.
    """
    logger.info(f"Loading labels from {labels_csv}")
    df_labels = pd.read_csv(labels_csv)
    
    tic_ids = df_labels['tic_id'].astype(str).tolist()
    logger.info(f"Fetching metadata for {len(tic_ids)} targets...")
    
    # In a real batch pipeline, we would query MAST TAP for the TIC table directly.
    # We reuse the existing tic_fetcher logic (astroquery Catalogs) for now.
    df_meta = fetch_tic_parameters(tic_ids)
    
    if df_meta.empty:
        logger.error("No metadata found.")
        return
        
    df_meta = df_meta.rename(columns={'ID': 'tic_id'})
    
    # Join labels and metadata
    df_labels['tic_id'] = df_labels['tic_id'].astype(str)
    df_meta['tic_id'] = df_meta['tic_id'].astype(str)
    
    df_merged = pd.merge(df_labels, df_meta, on='tic_id', how='inner')
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_merged.to_csv(output_csv, index=False)
    
    # Also save as parquet
    parquet_path = output_csv.replace(".csv", ".parquet")
    df_merged.to_parquet(parquet_path, index=False)
    
    logger.info(f"Saved {len(df_merged)} merged records to {output_csv} and {parquet_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../datasets/labels_raw.csv")
    parser.add_argument("--output", type=str, default="../datasets/metadata_merged.csv")
    args = parser.parse_args()
    fetch_metadata(args.input, args.output)
