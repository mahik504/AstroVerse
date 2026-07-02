import numpy as np
import pandas as pd
from astropy.stats import sigma_clip
from scipy.signal import savgol_filter
import lightkurve as lk

def fetch_lightcurve(tic_id, sector=None, author="SPOC"):
    """
    Fetches a light curve from MAST using lightkurve.
    """
    search_result = lk.search_lightcurve(tic_id, sector=sector, author=author)
    if len(search_result) == 0:
        raise ValueError(f"No light curve found for TIC {tic_id}")
    lc = search_result.download()
    return lc

def apply_quality_mask(lc):
    """
    Filters out bad data points using the quality flags provided in the FITS file.
    Uses the default 'default' bitmask from lightkurve which removes most severe anomalies.
    """
    return lc.remove_nans().remove_outliers(sigma_upper=5, sigma_lower=5)

def remove_outliers_sigma_clipping(time, flux, sigma=3.0):
    """
    Removes outliers using sigma clipping.
    """
    flux_arr = np.array(flux, dtype=float)
    filtered_data = sigma_clip(flux_arr, sigma=sigma, maxiters=5)
    
    if hasattr(filtered_data, 'mask') and filtered_data.mask is not np.ma.nomask:
        mask = ~filtered_data.mask
    else:
        mask = np.ones(len(flux_arr), dtype=bool)
        
    return np.array(time)[mask], flux_arr[mask]

def detrend_savgol(flux, window_length=101, polyorder=3):
    """
    Detrends the light curve using a Savitzky-Golay filter.
    Useful for removing long-term stellar variations while keeping sharp transit features.
    """
    trend = savgol_filter(flux, window_length, polyorder)
    detrended_flux = flux / trend
    return detrended_flux, trend

def normalize_flux(flux):
    """
    Normalizes flux by its median.
    """
    median_flux = np.nanmedian(flux)
    return flux / median_flux

def process_lightcurve(lc):
    """
    End-to-end preprocessing pipeline for a lightkurve LightCurve object.
    1. Apply quality mask
    2. Normalize
    3. Detrend
    4. Remove remaining outliers
    """
  
    lc = lc.remove_nans()
    
    time = lc.time.value
    flux = lc.pdcsap_flux.value
    
  
    valid = ~np.isnan(flux)
    time = time[valid]
    flux = flux[valid]

    flux_norm = normalize_flux(flux)
    

    window_length = 151
    if len(flux_norm) < window_length:
        window_length = len(flux_norm) - 1 if len(flux_norm) % 2 == 0 else len(flux_norm)
        
  
    if window_length % 2 == 0:
        window_length -= 1
    if window_length > 3:
        flux_detrended, trend = detrend_savgol(flux_norm, window_length=window_length, polyorder=3)
    else:
        flux_detrended = flux_norm
        trend = np.ones_like(flux_norm)
        

    time_clean, flux_clean = remove_outliers_sigma_clipping(time, flux_detrended, sigma=5.0)
    
    return {
        'time': time_clean,
        'flux': flux_clean,
        'trend': trend
    }

if __name__ == "__main__":
    print("Preprocessing module loaded.")
  