import logging
import os
import time
from pathlib import Path

def setup_central_logging(script_name):
    date_str = time.strftime("%Y-%m-%d")
    log_dir = Path(f"../logs/{date_str}")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "run.log"
    
    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler(log_file)
        
        c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)
        
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
    return logger
