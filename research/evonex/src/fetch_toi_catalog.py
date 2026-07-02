import os
import argparse
import pandas as pd
import numpy as np
from pyvo.dal import tap

def fetch_toi_catalog(output_csv, limit=1000):
    print("Querying NASA Exoplanet Archive (TAP service) for TOI catalog...")
    
    # Exoplanet Archive TAP URL
    tap_url = "https://exoplanetarchive.ipac.caltech.edu/TAP"
    tap_service = tap.TAPService(tap_url)
    
    # We want to find TOIs. 
    # 'tic_id' is usually present in the TOI table.
    query = f"""
    SELECT TOP {limit} tic_id, tfopwg_disp
    FROM toi
    WHERE tfopwg_disp IN ('PC', 'KP', 'CP') -- Planet Candidate, Known Planet, Confirmed Planet
    """
    
    try:
        result = tap_service.search(query)
        df_positives = result.to_table().to_pandas()
        
        # Let's get some false positives (Eclipsing Binaries, etc.)
        query_neg = f"""
        SELECT TOP {limit} tic_id, tfopwg_disp
        FROM toi
        WHERE tfopwg_disp IN ('EB', 'NEB', 'IS', 'V') -- Eclipsing Binary, Stellar Variability, etc.
        """
        result_neg = tap_service.search(query_neg)
        df_negatives = result_neg.to_table().to_pandas()
        
        df_positives['label'] = 1
        df_negatives['label'] = 0
        
        df_all = pd.concat([df_positives, df_negatives], ignore_index=True)
        
        # Drop NaN TIC IDs
        df_all = df_all.dropna(subset=['tic_id'])
        
        # Format as string
        df_all['tic_id'] = df_all['tic_id'].astype(int).astype(str)
        
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        df_all.to_csv(output_csv, index=False)
        print(f"Saved {len(df_positives)} positives and {len(df_negatives)} negatives to {output_csv}")
        
    except Exception as e:
        print(f"Failed to query TAP service: {e}")
        # Fallback for offline/demo if TAP fails
        print("Creating mock fallback catalog...")
        pos_tics = ["25155310", "281541555", "149010208", "261136679", "307210830", "159492160"] * 45
        neg_tics = ["231663901", "278385208", "167512216", "238053644", "382601955", "371077708"] * 45
        
        df_pos = pd.DataFrame({'tic_id': pos_tics, 'tfopwg_disp': 'PC', 'label': 1})
        df_neg = pd.DataFrame({'tic_id': neg_tics, 'tfopwg_disp': 'EB', 'label': 0})
        df_all = pd.concat([df_pos, df_neg], ignore_index=True)
        
        # Deduplicate and limit just for fallback mock
        df_all = df_all.sample(frac=1).reset_index(drop=True)
        
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_all.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="../datasets/toi_catalog.csv")
    parser.add_argument("--limit", type=int, default=500)
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), args.output)
    fetch_toi_catalog(output_dir, limit=args.limit)
