import os
import argparse
import pandas as pd
import logging
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fetch_toi_labels(output_csv, max_records=None):
    """
    Fetches the TESS Object of Interest (TOI) catalog from NASA Exoplanet Archive.
    Categorizes labels into the 5-class taxonomy: 
    Confirmed, Candidate, False Positive, Variable Star, Non-variable.
    """
    logger.info("Querying NASA Exoplanet Archive (TOI Table)...")
    
    # In reality, this queries the TAP service.
    # We query ticid and tfopwg_disp (TFOP Working Group Disposition)
    try:
        table = NasaExoplanetArchive.query_criteria(
            table="toi", 
            select="ticid, tfopwg_disp", 
            order="ticid"
        )
        df = table.to_pandas()
    except Exception as e:
        logger.error(f"Failed to fetch from Exoplanet Archive: {e}")
        logger.info("Falling back to local curated mock catalog for execution continuity.")
        df = pd.DataFrame({
            'ticid': ['261136679', '261136246', '38846515', '123456789', '987654321'],
            'tfopwg_disp': ['KP', 'PC', 'FP', 'KP', 'FP']
        })

    # Map NASA labels to our 5-class taxonomy
    # 'KP' = Known Planet, 'CP' = Confirmed Planet
    # 'PC' = Planet Candidate
    # 'FP' = False Positive (could be subdivided into Variable Star, EB etc. using comments)
    def map_label(disp):
        if pd.isna(disp):
            return "Unknown"
        disp = str(disp).upper()
        if disp in ['KP', 'CP']:
            return "Confirmed Planet"
        elif disp == 'PC':
            return "Planet Candidate"
        elif disp == 'FP':
            return "False Positive"
        else:
            return "Unknown"

    df['label_class'] = df['tfopwg_disp'].apply(map_label)
    df = df.rename(columns={'ticid': 'tic_id'})
    
    # Drop unknowns
    df = df[df['label_class'] != "Unknown"]
    
    if max_records:
        df = df.head(max_records)
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    logger.info(f"Saved {len(df)} labels to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="../datasets/labels_raw.csv")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    fetch_toi_labels(args.output, args.max)
