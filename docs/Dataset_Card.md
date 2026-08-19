# Dataset Card: AstroVerse Curated TOI

## Dataset Overview
The AstroVerse Curated TOI dataset is a high-quality, pre-processed collection of exoplanet transit light curves derived from the NASA Transiting Exoplanet Survey Satellite (TESS). It is constructed by pulling validated TESS Objects of Interest (TOIs) from the NASA Exoplanet Archive and retrieving their corresponding Sector 1 Pre-search Data Conditioning (PDCSAP) light curves via the MAST API.

- **Source:** NASA Exoplanet Archive / MAST
- **Format:** HDF5 Tensor Cache
- **Primary Feature:** `lc_flux` (Phase-folded light curve, 2000 sequence length)
- **Secondary Feature:** `tic_metadata` (12 physical stellar parameters)

## Preprocessing Details
- **Pipeline Aggregation**: Data is fetched via MAST. We prioritize high-quality SPOC and TESS-SPOC pipelines but fall back to QLP, TGLC, and ELEANOR to preserve dataset size and avoid systemic biases. The exact distribution of pipelines is logged in `dataset_report.json` for each generated subset.
- **Detrending**: Savitzky-Golay filter applied to remove long-term stellar variability.
- **Normalization**: Flux normalized by median flux.
- **Phase-folding**: BLS (Box Least Squares) applied to fold transit signals.
- **Interpolation**: Linear interpolation to standardize sequence length to $L=2000$.

## Dataset Versions
1. `v1-curated-100`: A small, 100-target dataset intended for continuous integration testing, architecture validation, and hyperparameter sweep sanity checks.
2. `v2-curated-500`: A medium-scale 500-target dataset for local model training.
3. `v4-paper-10000` *(Planned)*: The definitive 10,000-target dataset that will form the basis of the publication results.

## Data Preprocessing Pipeline
1. **Fetch TOI Catalog**: Queries the ADQL TAP service for confirmed planets (Label 1) and false positives (Label 0).
2. **MAST Retrieval**: Uses `lightkurve` to download the PDCSAP flux for Sector 1.
3. **Quality Masking**: Removes cadences flagged by TESS mission quality bitmasks (e.g., cosmic rays, momentum dumps).
4. **BLS Phase Folding**: Uses the Box Least Squares (BLS) periodogram to detect the dominant transit period and folds the entire light curve into a single phase-aligned transit event.
5. **Standardization**: Binning and linear interpolation to exactly 2,000 time steps, followed by zero-mean, unit-variance normalization.

## Known Limitations and Biases
- **Single-Sector Bias**: The current pipeline only evaluates Sector 1, which biases the model against long-period exoplanets (>15 days).
- **Selection Bias**: The dataset is intrinsically biased toward the TESS observational strategy (favoring M-dwarfs and short-period orbits).
- **Grazing Eclipsing Binaries**: A known failure mode exists where grazing eclipsing binaries mimic planetary transits, though the `tic_metadata` (Physics MLP) is designed to mitigate this.
