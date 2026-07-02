import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

def detect_transit_bls(time, flux, flux_err=None, min_period=0.5, max_period=20.0):
    """
    Runs the Box Least Squares (BLS) algorithm to detect periodic transit dips.
    
    Parameters:
        time (array-like): Time values (e.g., in BJD).
        flux (array-like): Normalized, detrended flux values.
        flux_err (array-like): Flux errors (optional).
        min_period (float): Minimum orbital period to search (days).
        max_period (float): Maximum orbital period to search (days).
        
    Returns:
        dict: The best fit parameters extracted by BLS.
        object: The generated BLS periodogram model for visualization.
    """
 
    if flux_err is not None:
        model = BoxLeastSquares(t=time, y=flux, dy=flux_err)
    else:
        model = BoxLeastSquares(t=time, y=flux)

   
    periods = np.linspace(min_period, max_period, 10000)

    durations = np.linspace(0.01, 0.2, 10)


    results = model.power(periods, durations)


    best_index = np.argmax(results.power)
    best_period = results.period[best_index]
    best_t0 = results.transit_time[best_index]
    best_duration = results.duration[best_index]
    best_depth = results.depth[best_index]

  
    power_mean = np.mean(results.power)
    power_std = np.std(results.power)
    snr = (results.power[best_index] - power_mean) / power_std

    best_parameters = {
        'period_days': best_period,
        'transit_epoch': best_t0,
        'duration_hours': best_duration * 24.0,  
        'transit_depth_ppm': best_depth * 1e6,   
        'bls_snr': snr
    }

    return best_parameters, results

if __name__ == "__main__":
 
    print("Initializing BLS Detector Test...")
    

    t = np.linspace(0, 27, 5000)
  
    f = np.random.normal(1.0, 0.001, len(t))
    

    period_true = 4.5
    epoch_true = 1.2
    duration_true = 0.1
    depth_true = 0.01
    
    phase = (t - epoch_true) % period_true
    transit_mask = (phase < duration_true / 2) | (phase > period_true - duration_true / 2)
    f[transit_mask] -= depth_true
    
    print("Running BLS on synthetic data...")
    best_params, periodogram = detect_transit_bls(t, f, min_period=1.0, max_period=10.0)
    
    print("\n--- BLS Extraction Results ---")
    print(f"True Period: {period_true} | Detected: {best_params['period_days']:.3f} days")
    print(f"True Epoch : {epoch_true} | Detected: {best_params['transit_epoch']:.3f}")
    print(f"True Depth : {depth_true*1e6:.0f} ppm | Detected: {best_params['transit_depth_ppm']:.0f} ppm")
    print(f"BLS SNR    : {best_params['bls_snr']:.2f}")
    print("BLS Module test completed successfully!")
