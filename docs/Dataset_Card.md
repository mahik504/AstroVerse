# Dataset card

## What this repository ships

Two photometric collections, plus the code that builds a third.

**Reference library** (`demo/`). Eighteen series, each 2,000 rows of time and flux. Eleven are grouped as planet-like, five as false-positive morphologies, two as threshold cases. Three are archive-style light curves. Fifteen are synthetic series built to isolate one shape: a U, a V, a contact binary, a flare, a blend, a giant host, an overlap. The manuscript measures a peak-to-median depth and a noise-referenced signal-to-noise ratio for every file.

**Processed cache.** Six TESS SPOC light curves, each stored as a flux vector of length 2,000 and twelve TIC columns. Stellar magnitude, temperature, and radius for those six are in the manuscript, with a central-transit equivalent radius.

**Catalog run.** `research/evonex/configs/paper.yaml` names the training schedule and a period window of 0.5–20 days. A scored TOI sample from that schedule is not in this release.

## Reduction

The running reduction is `research/evonex/src/preprocessing.py`.

1. Drop non-finite samples. Prefer PDCSAP flux, then SAP, then the generic flux column.
2. Divide by the median.
3. Remove a Savitzky–Golay trend, window 151, polynomial order 3.
4. Sigma-clip at 5σ, at most five iterations.
5. Box least squares: 10,000 periods from 0.5 to 20 days, 10 durations from 0.01 to 0.2 days.
6. Fold on the winning period and linearly resample to 2,000 samples.

The twelve stellar columns, in order, are Tmag, Teff, logg, rad, mass, rho, lum, d, ebv, ra, dec, contratio. They are stored on the catalog scales.

## Limits

A single TESS sector is about 27 days. Periods outside 0.5–20 days are outside the search grid. A harmonic of the true period still produces a fold. Synthetic series are not a substitute for a held-out TOI catalog.
