"""
Preprocessing Pipeline Tests
Validates detrending, sigma-clipping, and BLS detection on synthetic data.
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocessing import normalize_flux, detrend_savgol, remove_outliers_sigma_clipping
from bls_detector import detect_transit_bls


class TestNormalization:
    def test_median_one(self):
        flux = np.array([0.98, 1.0, 1.02, 1.0, 0.99])
        normed = normalize_flux(flux)
        assert abs(np.median(normed) - 1.0) < 0.01

    def test_preserves_shape(self):
        flux = np.random.uniform(0.95, 1.05, 500)
        normed = normalize_flux(flux)
        assert normed.shape == flux.shape


class TestDetrending:
    def test_output_shape(self):
        flux = np.random.normal(1.0, 0.01, 500)
        detrended, trend = detrend_savgol(flux, window_length=101, polyorder=3)
        assert detrended.shape == flux.shape
        assert trend.shape == flux.shape

    def test_removes_linear_trend(self):
        x = np.linspace(0, 1, 500)
        flux = 1.0 + 0.1 * x + np.random.normal(0, 0.001, 500)
        detrended, _ = detrend_savgol(flux, window_length=101, polyorder=3)
        assert np.std(detrended) < np.std(flux)


class TestSigmaClipping:
    def test_removes_outliers(self):
        time = np.arange(100, dtype=float)
        flux = np.ones(100)
        flux[50] = 10.0
        t_clean, f_clean = remove_outliers_sigma_clipping(time, flux, sigma=3.0)
        assert len(f_clean) < len(flux)
        assert 10.0 not in f_clean

    def test_preserves_clean_data(self):
        time = np.arange(100, dtype=float)
        flux = np.random.normal(1.0, 0.001, 100)
        t_clean, f_clean = remove_outliers_sigma_clipping(time, flux, sigma=5.0)
        assert len(f_clean) == len(flux)


class TestBLSDetection:
    def test_detects_injected_transit(self):
        t = np.linspace(0, 27, 5000)
        f = np.random.normal(1.0, 0.001, len(t))

        period_true = 4.5
        epoch_true = 1.2
        duration_true = 0.1
        depth_true = 0.01

        phase = (t - epoch_true) % period_true
        transit_mask = (phase < duration_true / 2) | (phase > period_true - duration_true / 2)
        f[transit_mask] -= depth_true

        params, _ = detect_transit_bls(t, f, min_period=1.0, max_period=10.0)

        assert abs(params["period_days"] - period_true) < 0.1, f"Period: {params['period_days']} vs {period_true}"
        assert params["bls_snr"] > 5.0, f"SNR too low: {params['bls_snr']}"

    def test_returns_required_keys(self):
        t = np.linspace(0, 27, 1000)
        f = np.random.normal(1.0, 0.001, len(t))
        params, _ = detect_transit_bls(t, f)
        required = ["period_days", "transit_epoch", "duration_hours", "transit_depth_ppm", "bls_snr"]
        for key in required:
            assert key in params, f"Missing key: {key}"
