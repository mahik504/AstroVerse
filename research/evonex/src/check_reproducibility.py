import sys
import os
import json
import hashlib
import torch
import platform

def verify_reproducibility(dataset_dir, config_path):
    print("=" * 60)
    print("REPRODUCIBILITY CHECKLIST")
    print("=" * 60)
    
    missing_fields = []
    
    # Check dataset
    if not os.path.exists(os.path.join(dataset_dir, "tess_cache.h5")):
        missing_fields.append("Dataset (tess_cache.h5 missing)")
        
    if not os.path.exists(os.path.join(dataset_dir, "checksum_manifest.json")):
        missing_fields.append("Dataset Checksum Manifest")
        
    # Check config
    if not os.path.exists(config_path):
        missing_fields.append(f"Config File ({config_path})")
        
    # Check environment
    py_version = platform.python_version()
    torch_version = torch.__version__
    cuda_available = torch.cuda.is_available()
    
    print(f"Python Version: {py_version}")
    print(f"PyTorch Version: {torch_version}")
    print(f"CUDA Available: {cuda_available}")
    
    if len(missing_fields) > 0:
        print("\n[ERROR] Experiment marked incomplete due to missing fields:")
        for f in missing_fields:
            print(f"  - {f}")
        sys.exit(1)
        
    print("\n[PASS] All reproducibility requirements met.")
    return True

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python check_reproducibility.py <dataset_dir> <config_path>")
        sys.exit(1)
    verify_reproducibility(sys.argv[1], sys.argv[2])
