import logging
import json

logger = logging.getLogger(__name__)
import os
import random
from astroquery.mast import Catalogs
import pandas as pd

def fetch_curated_targets():
    logger.info("Initiating connection to NASA MAST API...")
    
    
    curated_tics = [
        "25155310",   # WASP-126 (Planet)
        "164726113",  # TOI-125 (Planet)
        "281541555",  # Eclipsing Binary / False Positive
        "141914082",  # Normal Star
        "261136679",  # Pi Mensae (Planet)
        "270810587",  # TOI-120 (Planet)
        "307210830",  # HD 219666 (Planet)
        "149603524",  # TOI-150 (Planet)
        "38846515",   # TOI-172 (Planet)
        "92352620",   # TOI-144 (Planet)
        "201205369",  # Normal Star
        "425937691",  # Normal Star
        "425933644",  # Normal Star
        "307210844",  # Normal Star
        "149603530",  # Normal Star
    ]
    
    logger.info(f"Fetching stellar astrophysics metadata for {len(curated_tics)} targets...")
    
    database = []
    
    try:
        catalog_data = Catalogs.query_criteria(catalog="Tic", ID=curated_tics)
        df = catalog_data.to_pandas()
        
        for index, row in df.iterrows():
            tic_id = str(row['ID'])
            is_known_planet = tic_id in ["25155310", "164726113", "261136679", "270810587", "307210830", "149603524", "38846515", "92352620"]
            
            target_data = {
                "tic_id": tic_id,
                "tmag": float(row['Tmag']) if pd.notna(row['Tmag']) else 12.0,
                "temperature": float(row['Teff']) if pd.notna(row['Teff']) else 5500.0,
                "mass": float(row['mass']) if pd.notna(row['mass']) else 1.0,
                "radius": float(row['rad']) if pd.notna(row['rad']) else 1.0,
                "distance_pc": float(row['d']) if pd.notna(row['d']) else 100.0,
                "status": "CONFIRMED_PLANET" if is_known_planet else ("ECLIPSING_BINARY" if tic_id == "281541555" else "UNCLASSIFIED"),
                "last_observed": "Sector 27" # Hardcoded for demo UI consistency
            }
            database.append(target_data)
            
    except Exception as e:
        logger.error(f"Error communicating with MAST: {e}")
        return

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(data_dir, exist_ok=True)
    
    json_path = os.path.join(data_dir, "targets.json")
    with open(json_path, 'w') as f:
        json.dump(database, f, indent=4)
        
    logger.info(f"✅ Successfully compiled V1 Target Database with {len(database)} stars.")
    logger.info(f"Database saved to {json_path}")

if __name__ == "__main__":
    fetch_curated_targets()
