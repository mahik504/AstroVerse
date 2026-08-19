import os
import argparse
import pandas as pd
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_audit(metadata_csv, output_dir):
    """
    Generates a dataset audit verifying balance, missing values, and leakage.
    Produces audit_report.md and audit_report.json.
    """
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(metadata_csv)
    
    # Audit Checks
    duplicates = len(df) - len(df['tic_id'].unique())
    missing_rate = df.isna().mean().mean()
    class_balance = df['label_class'].value_counts().to_dict()
    
    # Label Leakage: Does the same TIC ID have conflicting labels?
    label_leakage = False
    if duplicates > 0:
        dup_tics = df[df.duplicated('tic_id', keep=False)]
        for tic, group in dup_tics.groupby('tic_id'):
            if len(group['label_class'].unique()) > 1:
                label_leakage = True
                
    audit_passed = (duplicates == 0) and not label_leakage
    
    audit_json = {
        "duplicates": duplicates,
        "missing_rate": round(missing_rate, 4),
        "class_balance": class_balance,
        "label_leakage_detected": label_leakage,
        "audit_passed": audit_passed
    }
    
    json_path = os.path.join(output_dir, "audit_report.json")
    with open(json_path, 'w') as f:
        json.dump(audit_json, f, indent=4)
        
    md_path = os.path.join(output_dir, "audit_report.md")
    with open(md_path, 'w') as f:
        f.write("# Dataset Audit Report\n\n")
        f.write(f"- **Audit Passed**: {audit_passed}\n")
        f.write(f"- **Duplicates**: {duplicates}\n")
        f.write(f"- **Label Leakage Detected**: {label_leakage}\n")
        f.write(f"- **Missing Value Rate**: {missing_rate:.2%}\n")
        f.write("\n## Class Balance\n")
        for cls, count in class_balance.items():
            f.write(f"- {cls}: {count}\n")
            
    logger.info(f"Audit completed. Passed: {audit_passed}. Wrote to {output_dir}/audit_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    args = parser.parse_args()
    generate_audit(args.input, args.outdir)
