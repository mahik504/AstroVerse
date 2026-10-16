# Scope

October 2026.

The forward pass, the reduction, and the reference library are in this repository. A scored comparison on the TESS Objects of Interest catalog is not.

- The box-least-squares grid runs from 0.5 to 20 days. Series outside that window are still visible to the convolutional expert. The search will not return their periods.
- One TESS sector is about 27 days. A harmonic of the true period still folds into a series the network will read as if it were the orbit.
- Stellar columns are passed on the scales the TIC returns.
- The processed cache shipped here holds six SPOC light curves.
- The eighteen files in `demo/` are a morphological reference library: three archive-style series and fifteen synthetic series built to isolate one shape. Depths in the manuscript are measured from those files.

Grazing binaries, diluted background eclipses, flares, and dips on giant hosts are in that library because those are the cases a gate is supposed to separate. Separating them in a catalog score is the next measurement.
