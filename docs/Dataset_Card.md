# Dataset Card: STScI Exoplanet Candidate Target List (xCTL v08.01)

## Dataset Overview
- **Name:** Exoplanet Candidate Target List (xCTL)
- **Version:** v08.01
- **Source:** NASA / STScI MAST Archive
- **URL:** https://archive.stsci.edu/tess/tic_ctl.html
- **Format:** CSV (497 MB)
- **Primary Use:** Prioritizing TESS targets for exoplanet transit detection.

## Dataset Characteristics
The xCTL is a curated subset of the TESS Input Catalog (TIC). It filters billions of point sources down to targets that are statistically and physically capable of hosting detectable transiting exoplanets.

### Features Used by EvoNex
| Column | Description | Used By |
|---|---|---|
| `ID` | Unique STScI/TIC identifier | Pipeline key |
| `Tmag` | TESS magnitude (brightness) | Physics MLP |
| `Teff` | Effective temperature (K) | Physics MLP |
| `logg` | Surface gravity | Physics MLP |
| `mass` | Stellar mass (solar units) | Physics MLP |
| `rad` | Stellar radius (solar units) | Physics MLP |
| `rho` | Stellar density | Physics MLP |
| `lum` | Luminosity | Physics MLP |
| `d` | Distance (parsecs) | Physics MLP |
| `ebv` | Reddening | Physics MLP |
| `ra` | Right ascension | Context |
| `dec` | Declination | Context |
| `contratio` | Contamination ratio | Physics MLP |

### Filtering Criteria (AstroVerse Pipeline)
1. **Dwarf Stars Only:** `rad < 2.0` — excludes giant stars where transits are too shallow.
2. **Temperature Bounds:** `3000K < Teff < 7000K` — M, K, G, and late F-type stars.
3. **Brightness:** `Tmag < 13.0` — ensures sufficient photon count for high SNR.

## Known Limitations & Biases
- **Binary Contamination:** Unresolved eclipsing binaries remain in the catalog.
- **M-Dwarf Bias:** Deeper transit signatures relative to stellar size favor M-dwarfs.
- **Crowding:** High-density galactic plane regions suffer from background flux contamination.
- **Missing Data:** Many faint stars lack mass, radius, or temperature measurements in the TIC.

## Ethical & Scientific Considerations
- **Open Access:** Public domain under NASA/STScI data use guidelines.
- **Reproducibility:** All inferences map to immutable TIC IDs, enabling third-party verification.
