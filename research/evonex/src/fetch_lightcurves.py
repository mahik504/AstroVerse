import os
import argparse
import pandas as pd
import logging
import time
import random
import h5py
import lightkurve as lk

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

AUTHOR_PRIORITY = ['SPOC', 'TESS-SPOC', 'QLP', 'TGLC', 'ELEANOR']

def search_lightcurve_with_backoff(target_name, max_retries=4):
    base_delay = 2.0
    for attempt in range(max_retries + 1):
        try:
            return lk.search_lightcurve(target_name, mission="TESS")
        except Exception as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                sleep_time = delay + random.uniform(-delay*0.2, delay*0.2)
                logger.warning(f"Lightkurve search error ({e}). Retrying in {sleep_time:.2f}s...")
                time.sleep(sleep_time)
            else:
                raise e

def fetch_lightcurves(metadata_csv, hdf5_cache, max_records=None):
    """
    Reads the TIC IDs from metadata_merged.csv, queries MAST for the lightcurves,
    and caches the raw flux arrays into HDF5.
    """
    logger.info(f"Loading metadata from {metadata_csv}")
    df = pd.read_csv(metadata_csv)
    
    if max_records:
        df = df.head(max_records)
        
    os.makedirs(os.path.dirname(hdf5_cache), exist_ok=True)
    
    with h5py.File(hdf5_cache, 'a') as h5f:
        # Inject self-describing provenance metadata
        h5f.attrs['dataset_version'] = "v2-curated-500"
        h5f.attrs['protocol_version'] = "v1.0"
        h5f.attrs['creation_date'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            import torch, numpy, astropy
            h5f.attrs['python_version'] = os.sys.version.split()[0]
            h5f.attrs['torch_version'] = torch.__version__
            h5f.attrs['lightkurve_version'] = lk.__version__
            h5f.attrs['astropy_version'] = astropy.__version__
            h5f.attrs['numpy_version'] = numpy.__version__
        except Exception:
            pass
            
        for i, row in df.iterrows():
            tic_id = str(row['tic_id'])
            
            if tic_id in h5f:
                logger.info(f"Skipping TIC {tic_id} (Already cached).")
                continue
                
            target_name = f"TIC {tic_id}"
            logger.info(f"[{i+1}/{len(df)}] Downloading {target_name}...")
            
            try:
                search_result = search_lightcurve_with_backoff(target_name)
                
                if len(search_result) == 0:
                    logger.warning(f"No TESS observations for {target_name}")
                    continue
                
                # Author priority
                chosen_author = None
                authors_available = search_result.author
                selected_lc_idx = 0
                for pref_author in AUTHOR_PRIORITY:
                    if pref_author in authors_available:
                        selected_lc_idx = list(authors_available).index(pref_author)
                        chosen_author = pref_author
                        break
                
                if not chosen_author:
                    chosen_author = authors_available[0]
                    
                lc = search_result[selected_lc_idx].download()
                
                # Cache RAW time and flux without preprocessing.
                # Preprocessing is handled downstream by build_dataset.py
                group = h5f.create_group(str(tic_id))
                group.create_dataset('time', data=lc.time.value)
                group.create_dataset('flux', data=lc.flux.value)
                group.attrs['pipeline'] = chosen_author
                
            except Exception as e:
                logger.error(f"Error fetching {target_name}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../datasets/metadata_merged.csv")
    parser.add_argument("--output", type=str, default="../datasets/tess_cache_raw.h5")
    parser.add_argument("--max", type=int, default=None)
    args = parser.parse_args()
    fetch_lightcurves(args.input, args.output, args.max)
