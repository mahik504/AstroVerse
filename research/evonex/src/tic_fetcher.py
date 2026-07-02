import logging
import pandas as pd

logger = logging.getLogger(__name__)
from astroquery.mast import Catalogs

CRITICAL_TIC_COLUMNS = [
    'ID',            
    'Tmag',          
    'Teff',         
    'logg',         
    'rad',           
    'mass',          
    'rho',          
    'lum',          
    'd',             
    'ebv',          
    'ra',           
    'dec',           
    'contratio'     
]

def fetch_tic_parameters(tic_ids: list) -> pd.DataFrame:
    """
    Fetches the critical stellar parameters for a list of TIC IDs directly from the MAST API.
    This avoids downloading the massive 10GB TIC CSV files locally.
    
    Parameters:
        tic_ids (list): A list of strings or integers representing the TIC IDs (e.g., ["25155310"]).
    
    Returns:
        pd.DataFrame: A dataframe containing the selected stellar properties.
    """
    logger.info(f"Querying MAST API for {len(tic_ids)} targets...")
    
    
    tic_ids_str = [str(tid).replace("TIC", "").strip() for tid in tic_ids]
    
    try:
      
        catalog_data = Catalogs.query_criteria(catalog="Tic", ID=tic_ids_str)
        df = catalog_data.to_pandas()
        
       
        available_cols = [col for col in CRITICAL_TIC_COLUMNS if col in df.columns]
        df_filtered = df[available_cols]
        
        return df_filtered
    except Exception as e:
        logger.error(f"Error querying MAST API: {e}")
        return pd.DataFrame()

def preprocess_tic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values and normalizes the TIC parameters so they can be fed into a Neural Network.
    
    Pitfall: Many stars in the TIC are missing mass or radius. We must impute these gracefully 
    (e.g., median imputation) rather than dropping the rows, otherwise we lose valuable light curve data.
    """
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

if __name__ == "__main__":
  
    sample_ids = ["25155310", "279741379"]
    print("Fetching TIC Data for:", sample_ids)
    tic_df = fetch_tic_parameters(sample_ids)
    print("Raw TIC Data:")
    print(tic_df[['ID', 'Teff', 'rad', 'mass', 'contratio']].head())
    
    clean_df = preprocess_tic_features(tic_df)
    print("\nPreprocessed (Normalized) TIC Data:")
    print(clean_df[['ID', 'Teff', 'rad', 'mass', 'contratio']].head())
