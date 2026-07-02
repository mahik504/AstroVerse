import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

def generate_reliability_diagram(y_true, y_prob, model_name, output_dir):
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
    plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=model_name)
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title(f"Reliability Diagram: {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, f"{model_name}_reliability.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path

def analyze_failures(y_true, y_prob, tic_metadata, output_dir, model_name, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    
    # Identify indices
    fp_idx = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_idx = np.where((y_pred == 0) & (y_true == 1))[0]
    
    analysis = {
        "model": model_name,
        "false_positives": {
            "count": len(fp_idx),
            "cases": []
        },
        "false_negatives": {
            "count": len(fn_idx),
            "cases": []
        }
    }
    
    # For a real implementation, we would inspect the TIC metadata to classify
    # e.g., Teff > 7000K might be a giant star, deep dips might be EBs.
    for i in fp_idx:
        # Mock logic based on metadata (index 3 is typically radius, etc.)
        # Here we just dump the metadata for manual review later
        analysis["false_positives"]["cases"].append({
            "index": int(i),
            "confidence": float(y_prob[i]),
            "metadata_snippet": tic_metadata[i][:3].tolist() if len(tic_metadata) > i else []
        })
        
    for i in fn_idx:
        analysis["false_negatives"]["cases"].append({
            "index": int(i),
            "confidence": float(y_prob[i]),
            "metadata_snippet": tic_metadata[i][:3].tolist() if len(tic_metadata) > i else []
        })
        
    report_path = os.path.join(output_dir, f"{model_name}_failure_analysis.json")
    with open(report_path, "w") as f:
        json.dump(analysis, f, indent=4)
        
    return report_path

if __name__ == "__main__":
    # Mock data to demonstrate script works
    np.random.seed(42)
    y_true = np.random.randint(0, 2, 100)
    y_prob = np.clip(y_true + np.random.normal(0, 0.3, 100), 0, 1)
    mock_tic = np.random.rand(100, 12)
    
    out_dir = os.path.dirname(__file__)
    generate_reliability_diagram(y_true, y_prob, "EvoMoE_Demo", out_dir)
    analyze_failures(y_true, y_prob, mock_tic, out_dir, "EvoMoE_Demo")
    print(f"Calibration and failure analysis generated in {out_dir}")
