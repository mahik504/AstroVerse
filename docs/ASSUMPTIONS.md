# Methodological Assumptions

This document explicitly acknowledges the assumptions underlying the AstroVerse pipeline. If any of these assumptions are proven false, the validity of the models' conclusions may be compromised.

## 1. Preprocessing Completeness
**Assumption**: Transit signals remain detectable after our preprocessing pipeline (median filtering + Savitzky-Golay).
**Risk**: Overly aggressive detrending may flatten shallow transits (Earth-sized planets around Sun-like stars), effectively removing the signal before the model sees it.

## 2. Metadata Accuracy
**Assumption**: Stellar parameters retrieved from the TIC (Radius, Mass, Teff) are sufficiently accurate.
**Risk**: TIC parameters often have high uncertainties or are derived from empirical relations rather than spectroscopic follow-up. Errors in stellar radius directly impact the model's physical validation of the transit depth.

## 3. Ground Truth Labels
**Assumption**: The NASA Exoplanet Archive labels (Confirmed, False Positive) are treated as absolute ground truth.
**Risk**: Historical "Confirmed" planets are occasionally retracted, and "False Positives" are sometimes rescued by better data. Label noise is inherently present.

## 4. Missing Observations
**Assumption**: Missing cadences (data gaps from satellite downlinks, cosmic ray momentum dumps) occur randomly and do not correlate with transit events.
**Risk**: If missing data correlates systematically with specific orbital phases, the phase-folding algorithm may produce artifact-heavy curves.

## 5. Temporal Stationary
**Assumption**: The properties of transits do not change significantly across TESS sectors.
**Risk**: Stellar activity (starspots, flares) evolves over time, which may alter the out-of-transit baseline differently in Sector 1 versus Sector 60.
