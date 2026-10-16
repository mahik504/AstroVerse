import os
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt

def generate_paper_figures(exp_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Plot Training Loss
    metrics_path = os.path.join(exp_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
        epochs = [m["epoch"] for m in metrics]
        losses = [m["loss"] for m in metrics]
        if not epochs or not all(np.isfinite(losses)):
            print("Skipped loss curve: metrics are missing or non-finite.")
            return

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, losses, marker='o', linestyle='-', color='b', label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('EvoMoE Training Loss over Epochs')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "fig_loss_curve.png"), dpi=300)
        plt.close()
        print("Generated: fig_loss_curve.png")
    
    # ROC, precision-recall, and gate histograms are drawn from a real
    # evaluation file when one exists. This script does not invent them.
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True, help="Path to experiment directory")
    args = parser.parse_args()
    
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "paper", "figures")
    generate_paper_figures(args.experiment, out_dir)
    print("Paper figures generation complete!")
