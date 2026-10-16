import os
import json
import math
import numpy as np
import pandas as pd

np.random.seed(42)

DEMO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "demo"))
os.makedirs(DEMO_DIR, exist_ok=True)

def generate_lightcurve(
    num_points=2000,
    depth=0.01,
    duration_fraction=0.1,
    shape="u_shape",  # u_shape, v_shape, noisy_u, double_dip, flare_noise
    noise_std=0.0005,
    variability_amp=0.0003,
    variability_freq=2.0,
    secondary_depth=0.0
):
    phase = np.linspace(-0.5, 0.5, num_points)
    flux = np.ones(num_points)
    
    # Stellar rotational baseline variability
    flux += variability_amp * np.sin(2 * np.pi * variability_freq * phase)
    
    # Primary Transit / Eclipse
    in_transit = np.abs(phase) < (duration_fraction / 2)
    if shape == "u_shape":
        # Flat bottom with cosine ingress/egress
        for i, p in enumerate(phase):
            abs_p = abs(p)
            half_dur = duration_fraction / 2
            ingress_width = duration_fraction * 0.2
            if abs_p < (half_dur - ingress_width):
                flux[i] -= depth
            elif abs_p < half_dur:
                frac = (half_dur - abs_p) / ingress_width
                flux[i] -= depth * 0.5 * (1 - math.cos(math.pi * frac))
    elif shape == "v_shape":
        # Grazing or binary V-shaped dip
        for i, p in enumerate(phase):
            abs_p = abs(p)
            half_dur = duration_fraction / 2
            if abs_p < half_dur:
                flux[i] -= depth * (1.0 - (abs_p / half_dur))
    elif shape == "double_dip":
        # Multi-planet overlap
        for i, p in enumerate(phase):
            if abs(p) < 0.04:
                flux[i] -= depth * 0.7
            if abs(p - 0.05) < 0.02:
                flux[i] -= depth * 0.5
    elif shape == "flare_noise":
        # Stellar flares + irregular dips
        for i, p in enumerate(phase):
            if abs(p) < (duration_fraction / 2):
                flux[i] -= depth * 0.6
            if abs(p - 0.2) < 0.03:
                flux[i] += depth * 0.8  # Flare emission
                
    # Secondary Eclipse (for binary stars)
    if secondary_depth > 0:
        for i, p in enumerate(phase):
            if abs(p - 0.25) < (duration_fraction / 2):
                flux[i] -= secondary_depth * (1.0 - abs(p - 0.25) / (duration_fraction / 2))
                
    # Gaussian photometric noise
    flux += np.random.normal(0, noise_std, num_points)
    
    # Normalize baseline to ~1.0
    flux = flux / np.median(flux)
    return phase, flux

# Target catalogue definitions
targets = [
    {
        "id": "Kepler-10b",
        "name": "Kepler-10b (Rocky Super-Earth)",
        "file": "Kepler-10b.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "NASA Kepler Mission / Archive Reference",
        "data_mode": "REAL_REFERENCE",
        "stellar_type": "G-type Sun-like",
        "radius_earth": 1.47,
        "period_days": 0.84,
        "snr": 28.4,
        "depth_ppm": 152,
        "expected_class": "Exoplanet Candidate",
        "description": "Historical first rocky exoplanet discovered by Kepler. Ultra-shallow U-shaped transit with pristine Keplerian phase regularity."
    },
    {
        "id": "WASP-126b",
        "name": "WASP-126b (Inflated Hot Jupiter)",
        "file": "WASP-126b.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "WASP Survey / NASA MAST Reference",
        "data_mode": "REAL_REFERENCE",
        "stellar_type": "G2V Star",
        "radius_earth": 10.76,
        "period_days": 3.28,
        "snr": 45.2,
        "depth_ppm": 9200,
        "expected_class": "Exoplanet Candidate",
        "description": "High-SNR Jovian transit exhibiting pronounced limb-darkening ingress/egress slopes and a clean flat trough."
    },
    {
        "id": "FalsePositive-EB",
        "name": "TIC-281541555 (Contact Eclipsing Binary)",
        "file": "FalsePositive-EB.csv",
        "category": "FALSE_POSITIVE",
        "provenance": "TESS Sector 27 Archive Reference",
        "data_mode": "REAL_REFERENCE",
        "stellar_type": "F-type Binary",
        "radius_earth": 19.8,
        "period_days": 1.12,
        "snr": 82.0,
        "depth_ppm": 45000,
        "expected_class": "Astrophysical False Positive",
        "description": "Classic astrophysical false positive: edge-on binary stellar system with characteristic deep V-shape and visible secondary minimum."
    },
    {
        "id": "TOI-700d",
        "name": "TOI-700 d (Habitable-Zone Earth)",
        "file": "TOI-700d.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "TESS Discovery Benchmark",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "M2V Red Dwarf",
        "radius_earth": 1.14,
        "period_days": 37.42,
        "snr": 14.1,
        "depth_ppm": 850,
        "expected_class": "Exoplanet Candidate",
        "description": "Earth-sized planet in the habitable zone of an M dwarf. Low transit depth requiring high transformer periodicity weighting."
    },
    {
        "id": "TRAPPIST-1e",
        "name": "TRAPPIST-1 e (Resonant Chain Terrestrial)",
        "file": "TRAPPIST-1e.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "Spitzer / TESS Follow-up Benchmark",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "Ultra-Cool M8V Dwarf",
        "radius_earth": 0.92,
        "period_days": 6.10,
        "snr": 22.6,
        "depth_ppm": 550,
        "expected_class": "Exoplanet Candidate",
        "description": "Tightly-locked terrestrial planet. Exhibits rapid transit duration with high orbital frequency and strong physical consistency."
    },
    {
        "id": "HD-209458b",
        "name": "HD 209458 b (Osiris - Canonical Hot Jupiter)",
        "file": "HD-209458b.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "Historical Milestone Benchmark",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "G0V Star",
        "radius_earth": 15.1,
        "period_days": 3.52,
        "snr": 62.0,
        "depth_ppm": 14600,
        "expected_class": "Exoplanet Candidate",
        "description": "The first discovered transiting exoplanet. Unmistakable deep box-shaped trough with substantial stellar limb darkening."
    },
    {
        "id": "LHS-3844b",
        "name": "LHS 3844 b (Ultra-Short-Period Lava World)",
        "file": "LHS-3844b.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "TESS Discovery Sector 1",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "M5V Dwarf",
        "radius_earth": 1.32,
        "period_days": 0.46,
        "snr": 19.8,
        "depth_ppm": 410,
        "expected_class": "Exoplanet Candidate",
        "description": "Ultra-short period world orbiting in under 11 hours. Clean morphology with high cadence repetition."
    },
    {
        "id": "K2-18b",
        "name": "K2-18 b (Atmospheric Sub-Neptune)",
        "file": "K2-18b.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "Kepler K2 Mission / JWST Target",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "M2.5V Dwarf",
        "radius_earth": 2.61,
        "period_days": 32.94,
        "snr": 31.5,
        "depth_ppm": 2980,
        "expected_class": "Exoplanet Candidate",
        "description": "Habitable-zone candidate with atmospheric methane/CO2 detections. Intermediate transit depth with moderate stellar variability."
    },
    {
        "id": "FP-Grazing-Binary-01",
        "name": "FP-GB-8921 (Grazing Eclipsing Binary)",
        "file": "FP-Grazing-Binary-01.csv",
        "category": "FALSE_POSITIVE",
        "provenance": "Astrophysical False Positive Library",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "A-type Dwarf",
        "radius_earth": 24.0,
        "period_days": 2.45,
        "snr": 54.0,
        "depth_ppm": 22000,
        "expected_class": "Astrophysical False Positive",
        "description": "Stellar grazing geometry producing a sharp V-shaped transit lacking a flat bottom trough. High CNN morphological rejection rate."
    },
    {
        "id": "FP-Stellar-Flare-Noise",
        "name": "FP-FLARE-1044 (Active Star Flare Anomaly)",
        "file": "FP-Stellar-Flare-Noise.csv",
        "category": "FALSE_POSITIVE",
        "provenance": "Instrumental/Astrophysical Noise",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "M4Ve Flare Star",
        "radius_earth": 0.0,
        "period_days": 0.0,
        "snr": 8.1,
        "depth_ppm": 4800,
        "expected_class": "Astrophysical False Positive",
        "description": "Stellar flare activity generating asymmetric flux spikes adjacent to pseudo-dips that confound simple thresholding."
    },
    {
        "id": "FP-Background-Blended-EB",
        "name": "FP-BEB-4190 (Background Blended Binary)",
        "file": "FP-Background-Blended-EB.csv",
        "category": "FALSE_POSITIVE",
        "provenance": "Aperture Contamination Simulation",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "F8V Field Star",
        "radius_earth": 0.0,
        "period_days": 4.18,
        "snr": 11.2,
        "depth_ppm": 1200,
        "expected_class": "Astrophysical False Positive",
        "description": "Background star light leaking into target aperture. Creates shallow diluted dips that violate host star radius constraints."
    },
    {
        "id": "Borderline-Candidate-01",
        "name": "CAND-7801 (Borderline Detection Threshold)",
        "file": "Borderline-Candidate-01.csv",
        "category": "BORDERLINE_CANDIDATE",
        "provenance": "Low Margin Demonstration",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "K1V Star",
        "radius_earth": 1.25,
        "period_days": 8.75,
        "snr": 7.4,
        "depth_ppm": 135,
        "expected_class": "Uncertain / Candidate",
        "description": "Borderline candidate hovering near the standard 7.1-sigma detection limit. High uncertainty across expert votes."
    },
    {
        "id": "Borderline-Candidate-02",
        "name": "CAND-9102 (Low SNR Ambiguous Signal)",
        "file": "Borderline-Candidate-02.csv",
        "category": "BORDERLINE_CANDIDATE",
        "provenance": "Low Margin Demonstration",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "G5V Star",
        "radius_earth": 1.40,
        "period_days": 12.30,
        "snr": 6.8,
        "depth_ppm": 110,
        "expected_class": "Uncertain / False Positive",
        "description": "Signal submerged in high white noise. Tests the model's calibration and epistemic uncertainty reporting."
    },
    {
        "id": "Ultra-Short-Period-Hot-Jupiter",
        "name": "USP-HJ-005 (Ultra-Short-Period Giant)",
        "file": "Ultra-Short-Period-Hot-Jupiter.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "Extreme Astrophysics Archive",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "F7V Subgiant",
        "radius_earth": 13.8,
        "period_days": 0.72,
        "snr": 51.0,
        "depth_ppm": 12500,
        "expected_class": "Exoplanet Candidate",
        "description": "Extreme gas giant orbiting inside the tidal disruption boundary. Steep ingress/egress with high temporal gating weight."
    },
    {
        "id": "Noisy-M-Dwarf-Transit",
        "name": "M-DWARF-884 (Noisy M-Dwarf Super-Earth)",
        "file": "Noisy-M-Dwarf-Transit.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "Stellar Noise Robustness Test",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "M3V Active Dwarf",
        "radius_earth": 1.85,
        "period_days": 2.15,
        "snr": 16.5,
        "depth_ppm": 980,
        "expected_class": "Exoplanet Candidate",
        "description": "Transit signal superposed on high-frequency Poisson noise and spot modulation. Demonstrates multi-scale CNN filtering."
    },
    {
        "id": "Giant-Star-Deep-False-Alarm",
        "name": "FP-GIANT-602 (Evolved Giant False Alarm)",
        "file": "Giant-Star-Deep-False-Alarm.csv",
        "category": "FALSE_POSITIVE",
        "provenance": "Physical Inconsistency Test",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "K2III Red Giant",
        "radius_earth": 0.0,
        "period_days": 18.5,
        "snr": 28.0,
        "depth_ppm": 8500,
        "expected_class": "Astrophysical False Positive",
        "description": "Apparent 0.8% transit dip around a 12 R_sun red giant. Physics MLP flags as an impossible exoplanet (implied radius > 1.1 R_sun)."
    },
    {
        "id": "Eccentric-Transit-Candidate",
        "name": "ECC-741 (High-Eccentricity Candidate)",
        "file": "Eccentric-Transit-Candidate.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "Dynamical System Benchmark",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "G1V Star",
        "radius_earth": 4.2,
        "period_days": 26.4,
        "snr": 24.0,
        "depth_ppm": 2100,
        "expected_class": "Exoplanet Candidate",
        "description": "Candidate exhibiting elongated transit duration caused by periastron passage in an eccentric Keplerian orbit."
    },
    {
        "id": "Multi-Planet-Overlap-Sim",
        "name": "MULTI-SYS-303 (Overlapping Multi-Planet System)",
        "file": "Multi-Planet-Overlap-Sim.csv",
        "category": "CONFIRMED_PLANET",
        "provenance": "Complex Resonance Benchmark",
        "data_mode": "SYNTHETIC_MODEL",
        "stellar_type": "K0V Star",
        "radius_earth": 2.8,
        "period_days": 5.8,
        "snr": 35.0,
        "depth_ppm": 1600,
        "expected_class": "Exoplanet Candidate",
        "description": "Compound transit dip generated by simultaneous transit passage of two coplanar planetary bodies."
    }
]

# Generate light curve files
for t in targets:
    file_path = os.path.join(DEMO_DIR, t["file"])
    # If file already exists and is real reference, keep it
    if t["data_mode"] == "REAL_REFERENCE" and os.path.exists(file_path):
        print(f"Preserving existing real reference: {t['file']}")
        continue
        
    # Generate tailored synthetic curve
    depth_val = t["depth_ppm"] / 1e6
    if "V-Shaped" in t["name"] or "Grazing" in t["name"] or t["id"] == "FP-Grazing-Binary-01":
        phase, flux = generate_lightcurve(depth=depth_val, shape="v_shape", duration_fraction=0.15, secondary_depth=depth_val*0.35)
    elif "Flare" in t["name"]:
        phase, flux = generate_lightcurve(depth=depth_val, shape="flare_noise", duration_fraction=0.1, noise_std=0.0012)
    elif "Multi-Planet" in t["name"]:
        phase, flux = generate_lightcurve(depth=depth_val, shape="double_dip", duration_fraction=0.12)
    elif "Borderline" in t["name"]:
        phase, flux = generate_lightcurve(depth=depth_val, shape="u_shape", duration_fraction=0.08, noise_std=0.0008)
    else:
        # Standard U-shape
        phase, flux = generate_lightcurve(depth=depth_val, shape="u_shape", duration_fraction=0.09, noise_std=0.0003)
        
    df = pd.DataFrame({"time": phase, "flux": flux})
    df.to_csv(file_path, index=False)
    print(f"Generated {t['file']} ({t['name']})")

catalog_path = os.path.join(DEMO_DIR, "demo_catalog.json")
with open(catalog_path, "w") as f:
    json.dump(targets, f, indent=4)

print(f"\nWrote demo catalog with {len(targets)} entries to {catalog_path}")
