import os
import json
import logging
import argparse
from datetime import datetime
import h5py
import numpy as np
import pandas as pd
import lightkurve as lk
from scipy.interpolate import interp1d

from preprocessing import process_lightcurve
from bls_detector import detect_transit_bls
from tic_fetcher import fetch_tic_parameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Sample TOIs (Positives) and False Positives (Negatives) for demo dataset
DEMO_TICS = {
    "positives": ["25155310", "281541555", "149010208", "261136679", "307210830", "159492160"],
    "negatives": ["231663901", "278385208", "167512216", "238053644", "382601955", "371077708"]
}

def interpolate_lc(phase, flux, seq_length=2000):
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]
    f_interp = interp1d(phase_sorted, flux_sorted, kind='linear', fill_value="extrapolate")
    fixed_phase = np.linspace(phase_sorted.min(), phase_sorted.max(), seq_length)
    fixed_flux = f_interp(fixed_phase)
    return fixed_flux

def build_dataset(version_name, output_dir, tics_pos=None, tics_neg=None, catalog_csv=None, max_samples=None):
    os.makedirs(output_dir, exist_ok=True)
    h5_path = os.path.join(output_dir, "tess_cache.h5")
    
    total_processed = 0
    success_pos = 0
    success_neg = 0

    if catalog_csv and os.path.exists(catalog_csv):
        logger.info(f"Loading targets from {catalog_csv}")
        df = pd.read_csv(catalog_csv)
        if max_samples:
            df = df.sample(n=min(len(df), max_samples), random_state=42)
        all_tics = [(str(row['tic_id']), int(row['label'])) for _, row in df.iterrows()]
    else:
        logger.info("Using explicit TIC lists (fallback)")
        all_tics = [(tic, 1) for tic in tics_pos] + [(tic, 0) for tic in tics_neg]
        if max_samples:
            # We can't really shuffle nicely here, just taking first N
            all_tics = all_tics[:max_samples]
    
    with h5py.File(h5_path, 'w') as h5f:
        for tic_id, label in all_tics:
            target_name = f"TIC {tic_id}"
            logger.info(f"Processing {target_name} (Label: {label})...")
            
            try:
                search_result = lk.search_lightcurve(target_name, author="SPOC")
                if len(search_result) == 0:
                    logger.warning(f"No SPOC lightcurve found for {target_name}")
                    continue
                
                lc = search_result[0].download()
                clean_data = process_lightcurve(lc)
                t = clean_data['time']
                f = clean_data['flux']
                
                # QA/QC
                if np.isnan(f).any():
                    logger.warning(f"NaNs present after preprocessing for {target_name}. Skipping.")
                    continue
                if len(t) < 500:
                    logger.warning(f"Lightcurve too short for {target_name} ({len(t)} pts). Skipping.")
                    continue
                
                best_params, _ = detect_transit_bls(t, f)
                period = best_params['period_days']
                t0 = best_params['transit_epoch']
                phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
                
                fixed_flux = interpolate_lc(phase, f)
                
                df_tic = fetch_tic_parameters([tic_id])
                tic_features = df_tic.drop(columns=['ID']).values[0]
                
                if str(tic_id) in h5f:
                    logger.warning(f"TIC {tic_id} already exists in cache. Skipping duplicate.")
                    continue
                    
                # Save to HDF5
                group = h5f.create_group(str(tic_id))
                group.create_dataset('lc_flux', data=fixed_flux)
                group.create_dataset('tic_metadata', data=tic_features)
                group.attrs['label'] = label
                
                total_processed += 1
                if label == 1:
                    success_pos += 1
                else:
                    success_neg += 1
                    
            except Exception as e:
                logger.error(f"Error processing {target_name}: {e}")

    # Generate metadata.json
    metadata = {
        "dataset_version": version_name,
        "creation_date": datetime.utcnow().isoformat() + "Z",
        "source": "NASA MAST API (lightkurve)",
        "total_targets_attempted": len(all_tics),
        "total_targets_saved": total_processed,
        "positives": success_pos,
        "negatives": success_neg,
        "filtering_rules": [
            "Requires SPOC author",
            "Must pass NaNs check post-preprocessing",
            "Minimum 500 valid observation points"
        ],
        "random_seed": 42
    }
    
    with open(os.path.join(output_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=4)
        
    logger.info(f"Dataset {version_name} built successfully! Saved {total_processed} targets.")
    
    # Generate statistics
    try:
        from dataset_stats import generate_dataset_statistics
        logger.info("Generating dataset statistics...")
        generate_dataset_statistics(output_dir)
    except Exception as e:
        logger.error(f"Failed to generate dataset statistics: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="v1-curated", help="Dataset version name")
    parser.add_argument("--catalog", type=str, default=None, help="Path to TOI catalog CSV")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples to process")
    args = parser.parse_args()
    
    output_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", args.version)
    logger.info(f"Building dataset: {args.version} -> {output_dir}")
    
    # Set fixed seed for reproducibility
    np.random.seed(42)
    
    catalog_path = None
    if args.catalog:
        catalog_path = os.path.join(os.path.dirname(__file__), args.catalog)
        
    build_dataset(args.version, output_dir, DEMO_TICS["positives"], DEMO_TICS["negatives"], catalog_csv=catalog_path, max_samples=args.max_samples)
