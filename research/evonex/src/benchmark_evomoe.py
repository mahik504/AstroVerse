"""
EvoMoE Benchmark Gate & Evaluation
Executes baseline models ONLY if strict scientific criteria are met.
"""
import os
import json
import sys
import pandas as pd

def check_benchmark_gate(dataset_dir):
    """
    Enforces the Benchmark Gate criteria (AstroVerse V6 Phase 5).
    """
    print("=" * 60)
    print("ASTROVERSE V6 BENCHMARK GATE")
    print("=" * 60)
    
    report_path = os.path.join(dataset_dir, "dataset_report.json")
    if not os.path.exists(report_path):
        print("[FAIL] dataset_report.json not found.")
        sys.exit(1)
        
    with open(report_path, "r") as f:
        report = json.load(f)
        
    total_targets = report.get("total_successful", 0)
    positives = report.get("positives", 0)
    negatives = report.get("negatives", 0)
    
    if total_targets < 500:
        print(f"[FAIL] Minimum dataset size not met. Required: 500. Found: {total_targets}")
        sys.exit(1)
        
    if positives < 50 or negatives < 50:
        print(f"[FAIL] Minimum positive/negative samples not met. Found Pos: {positives}, Neg: {negatives}")
        sys.exit(1)
        
    checksum_path = os.path.join(dataset_dir, "checksum_manifest.json")
    if not os.path.exists(checksum_path):
        print("[FAIL] Reproducibility manifest (checksum_manifest.json) not found.")
        sys.exit(1)
        
    check_dataset_audit(dataset_dir)
        
    duplicate_path = os.path.join(dataset_dir, "duplicate_report.csv")
    if os.path.exists(duplicate_path):
        dups = pd.read_csv(duplicate_path)
        if not dups.empty:
            print("[WARN] Dataset contains duplicates before generation. Make sure they were dropped.")
            
    print("[PASS] Benchmark Gate cleared. Execution authorized.")
    return True

def check_dataset_audit(dataset_dir):
    """
    Binary Benchmark Gate: Reads dataset_audit.json. 
    If audit_passed is False, or file is missing, the experiment aborts.
    """
    audit_path = Path(dataset_dir) / "audit_report.json"
    if not audit_path.exists():
        logger.error(f"BENCHMARK ABORTED: Audit report missing at {audit_path}. Run dataset_audit.py first.")
        sys.exit(1)
        
    with open(audit_path, 'r') as f:
        audit = json.load(f)
        
    if not audit.get('audit_passed', False):
        logger.error(f"BENCHMARK ABORTED: Dataset Audit failed. Review {audit_path} for label leakage or duplicates.")
        sys.exit(1)
        
    logger.info("Dataset Audit passed. Proceeding with benchmark.")

def run_benchmarks(dataset_dir=None):
    if dataset_dir:
        check_benchmark_gate(dataset_dir)
    else:
        print("[FAIL] Dataset directory not provided.")
        sys.exit(1)
        
    # Placeholder for baseline execution
    print("Benchmarking framework is ready for large-scale evaluation.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_benchmarks(sys.argv[1])
    else:
        print("Usage: python benchmark_evomoe.py <dataset_dir>")
        sys.exit(1)
