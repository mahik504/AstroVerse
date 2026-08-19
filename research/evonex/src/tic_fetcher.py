import logging
import pandas as pd
import time
import random
from astroquery.mast import Catalogs
import requests.exceptions

logger = logging.getLogger(__name__)

CRITICAL_TIC_COLUMNS = [
    'ID', 'Tmag', 'Teff', 'logg', 'rad', 'mass', 'rho', 'lum', 'd', 'ebv', 'ra', 'dec', 'contratio'     
]

def fetch_tic_parameters(tic_ids: list, max_retries=4) -> pd.DataFrame:
    """
    Fetches the critical stellar parameters for a list of TIC IDs directly from the MAST API
    with randomized exponential backoff for robustness against rate limits.
    """
    logger.info(f"Querying MAST API for {len(tic_ids)} targets...")
    tic_ids_str = [str(tid).replace("TIC", "").strip() for tid in tic_ids]
    
    base_delay = 2.0
    
    for attempt in range(max_retries + 1):
        try:
            catalog_data = Catalogs.query_criteria(catalog="Tic", ID=tic_ids_str)
            df = catalog_data.to_pandas()
            available_cols = [col for col in CRITICAL_TIC_COLUMNS if col in df.columns]
            return df[available_cols]
            
        except Exception as e:
            if attempt < max_retries:
                # Randomized exponential backoff (e.g. 2s, 4s, 8s, 16s with +/- 20% jitter)
                delay = base_delay * (2 ** attempt)
                jitter = delay * 0.2
                sleep_time = delay + random.uniform(-jitter, jitter)
                
                logger.warning(f"MAST API error ({e}). Retrying in {sleep_time:.2f}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to fetch TIC parameters after {max_retries} retries: {e}")
                return pd.DataFrame()

def preprocess_tic_features(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    
    for col in df_clean.columns:
        if col != 'ID':
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
            
    for col in df_clean.columns:
        if col != 'ID':
            mean = df_clean[col].mean()
            std = df_clean[col].std()
            if std > 0:
                df_clean[col] = (df_clean[col] - mean) / std
            else:
                df_clean[col] = 0.0
                
    return df_clean
