import os
import argparse
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def generate_dataset_statistics(version_dir):
    h5_path = os.path.join(version_dir, "tess_cache.h5")
    if not os.path.exists(h5_path):
        print(f"Error: {h5_path} not found.")
        return

    labels = []
    temperatures = []
    radii = []
    masses = []
    
    with h5py.File(h5_path, 'r') as h5f:
        tics = list(h5f.keys())
        for tic in tics:
            group = h5f[tic]
            labels.append(group.attrs['label'])
            
            # TIC metadata is [Teff, Mass, Radius, ...] assuming order from tic_fetcher.py
            # For simplicity, we just extract the first 3 if they exist.
            # Assuming: 0: Teff, 1: logg, 2: MH, 3: Rad, 4: Mass, 5: rho
            meta = group['tic_metadata'][:]
            if len(meta) >= 5:
                temperatures.append(meta[0])
                radii.append(meta[3])
                masses.append(meta[4])

    # 1. Label Distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x=labels, palette="Set2")
    plt.title("Label Distribution (0 = False Positive, 1 = Exoplanet)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.savefig(os.path.join(version_dir, "label_distribution.png"), dpi=150)
    plt.close()

    # 2. Stellar Temperature Distribution
    plt.figure(figsize=(6, 4))
    valid_teff = [t for t in temperatures if not np.isnan(t)]
    if valid_teff:
        sns.histplot(valid_teff, bins=20, kde=True, color="orange")
        plt.title("Stellar Effective Temperature (T_eff) Distribution")
        plt.xlabel("Temperature (K)")
        plt.ylabel("Count")
        plt.savefig(os.path.join(version_dir, "stellar_teff_distribution.png"), dpi=150)
    plt.close()
    
    # 3. Stellar Radius vs Mass (Stellar Types)
    plt.figure(figsize=(6, 4))
    valid_idx = [i for i in range(len(radii)) if not np.isnan(radii[i]) and not np.isnan(masses[i])]
    if valid_idx:
        v_radii = [radii[i] for i in valid_idx]
        v_masses = [masses[i] for i in valid_idx]
        sns.scatterplot(x=v_masses, y=v_radii, hue=[labels[i] for i in valid_idx], palette="Set1")
        plt.title("Stellar Radius vs Mass")
        plt.xlabel("Mass (Solar Masses)")
        plt.ylabel("Radius (Solar Radii)")
        plt.savefig(os.path.join(version_dir, "stellar_type_distribution.png"), dpi=150)
    plt.close()

    # Generate Report JSON
    report = {
        "total_samples": len(labels),
        "positive_samples": sum(labels),
        "negative_samples": len(labels) - sum(labels),
        "teff_mean": np.nanmean(temperatures) if temperatures else None,
        "teff_std": np.nanstd(temperatures) if temperatures else None,
        "radius_mean": np.nanmean(radii) if radii else None,
        "mass_mean": np.nanmean(masses) if masses else None
    }

    with open(os.path.join(version_dir, "dataset_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    print(f"Dataset statistics generated at {version_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, required=True, help="Dataset version directory name")
    args = parser.parse_args()
    
    dataset_dir = os.path.join(os.path.dirname(__file__), "..", "datasets", args.version)
    generate_dataset_statistics(dataset_dir)
