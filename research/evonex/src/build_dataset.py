import os
import argparse
import yaml
import logging
import subprocess

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_script(module_name, args):
    cmd = ["python", "-m", module_name] + args
    logger.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def build_dataset(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)['dataset']
        
    logger.info(f"Building Dataset: {config['name']}")
    
    # 1. Fetch Labels
    run_script("src.fetch_labels", ["--output", config['labels_raw'], "--max", "5"])
    
    # 2. Fetch Metadata
    run_script("src.fetch_metadata", ["--input", config['labels_raw'], "--output", config['metadata_merged']])
    
    # 3. Fetch Lightcurves
    raw_cache = config['hdf5_cache'].replace(".h5", "_raw.h5")
    run_script("src.fetch_lightcurves", ["--input", config['metadata_merged'], "--output", raw_cache, "--max", "5"])
    
    # 4. Preprocess (this would be a separate script process_dataset.py, skipped here for brevity)
    logger.info(f"Dataset {config['name']} raw cache built at {raw_cache}.")
    logger.info("Next step: Run `python src/process_dataset.py` to apply v1 preprocessing.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    build_dataset(args.config)
