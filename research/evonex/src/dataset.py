import os
import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import lightkurve as lk
from scipy.interpolate import interp1d

from preprocessing import process_lightcurve
from bls_detector import detect_transit_bls
from tic_fetcher import fetch_tic_parameters

class TESSDataset(Dataset):
    """
    PyTorch DataLoader for TESS Exoplanet data.
    Features:
    1. HDF5 Caching: Saves processed tensors locally to avoid repeated MAST downloads.
    2. Phase-Folding: Uses BLS to fold the light curve at the detected transit period.
    3. Multimodal: Returns both 1D Light Curve tensors and 1D Stellar Metadata tensors.
    """
    def __init__(self, tic_ids, labels=None, cache_dir="../data/cache", seq_length=2000):
        self.tic_ids = tic_ids
        self.labels = labels
        self.cache_dir = cache_dir
        self.seq_length = seq_length

        os.makedirs(self.cache_dir, exist_ok=True)
        self.h5_path = os.path.join(self.cache_dir, "tess_cache.h5")

    def __len__(self):
        return len(self.tic_ids)

    def _interpolate_lc(self, phase, flux):
        """
        Interpolates the phase-folded light curve to a fixed length (seq_length).
        PyTorch requires all input tensors in a batch to be the exact same size.
        """
        sort_idx = np.argsort(phase)
        phase_sorted = phase[sort_idx]
        flux_sorted = flux[sort_idx]

        f_interp = interp1d(phase_sorted, flux_sorted, kind='linear', fill_value="extrapolate")

        fixed_phase = np.linspace(phase_sorted.min(), phase_sorted.max(), self.seq_length)
        fixed_flux = f_interp(fixed_phase)

        return fixed_flux

    def __getitem__(self, idx):
        tic_id = str(self.tic_ids[idx])

        target_name = f"TIC {tic_id}" if not tic_id.startswith("TIC") else tic_id
        clean_tic_id = target_name.replace("TIC ", "").strip()

        # Try reading from HDF5 cache (read-only mode for thread safety)
        try:
            with h5py.File(self.h5_path, 'r') as h5f:
                if clean_tic_id in h5f:
                    group = h5f[clean_tic_id]
                    lc_tensor = torch.tensor(group['lc_flux'][:], dtype=torch.float32)
                    tic_tensor = torch.tensor(group['tic_metadata'][:], dtype=torch.float32)
                    if self.labels is not None:
                        return lc_tensor, tic_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
                    return lc_tensor, tic_tensor
        except (OSError, KeyError):
            pass

        # Cache miss — download from MAST
        print(f"[{target_name}] Cache miss. Fetching from MAST API...")

        try:
            search_result = lk.search_lightcurve(target_name, author="SPOC")
            if len(search_result) == 0:
                raise ValueError(f"No light curve found for {target_name}")
            lc = search_result[0].download()
        except Exception as e:
            print(f"Error downloading {target_name}: {e}")
            lc_tensor = torch.zeros(self.seq_length, dtype=torch.float32)
            tic_tensor = torch.zeros(12, dtype=torch.float32)
            if self.labels is not None:
                return lc_tensor, tic_tensor, torch.tensor(self.labels[idx], dtype=torch.long)
            return lc_tensor, tic_tensor

        # Preprocess the raw lightcurve
        clean_data = process_lightcurve(lc)
        t = clean_data['time']
        f = clean_data['flux']

        # Phase-fold using BLS transit detection
        try:
            best_params, _ = detect_transit_bls(t, f)
            period = best_params['period_days']
            t0 = best_params['transit_epoch']
            phase = ((t - t0 + 0.5 * period) % period) - 0.5 * period
        except ValueError:
            phase = t

        # Interpolate to fixed sequence length
        fixed_flux = self._interpolate_lc(phase, f)
        lc_tensor = torch.tensor(fixed_flux, dtype=torch.float32)

        # Fetch stellar metadata from TIC
        df_tic = fetch_tic_parameters([clean_tic_id])
        tic_features = df_tic.drop(columns=['ID']).values[0]
        tic_tensor = torch.tensor(tic_features, dtype=torch.float32)

        # Write to HDF5 cache (append mode, only for writing)
        try:
            with h5py.File(self.h5_path, 'a') as h5f:
                if clean_tic_id not in h5f:
                    group = h5f.create_group(clean_tic_id)
                    group.create_dataset('lc_flux', data=fixed_flux)
                    group.create_dataset('tic_metadata', data=tic_features)
        except OSError:
            pass

        if self.labels is not None:
            label_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return lc_tensor, tic_tensor, label_tensor

        return lc_tensor, tic_tensor

if __name__ == "__main__":
    print("Testing TESSDataset Data Loader (with HDF5 Caching)...")

    sample_tics = ["25155310", "281541555"]
    sample_labels = [1, 0]
    dataset = TESSDataset(tic_ids=sample_tics, labels=sample_labels)

    print("\n--- FIRST PASS (Downloading) ---")
    lc1, tic1, label1 = dataset[0]
    print(f"TIC 25155310 -> LC Shape: {lc1.shape}, TIC Shape: {tic1.shape}, Label: {label1}")

    import time
    print("\n--- SECOND PASS (Cache Hit) ---")
    start = time.time()
    lc1_cached, tic1_cached, label1_cached = dataset[0]
    duration = time.time() - start
    print(f"TIC 25155310 Loaded from cache in {duration:.4f} seconds!")
    print("DataLoader test completed successfully.")
