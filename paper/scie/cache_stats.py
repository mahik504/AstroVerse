import h5py
import numpy as np

p = r"C:\projects\AstroVerse\research\evonex\datasets\v2-curated-500\tess_cache.h5"
RSUN_REARTH = 109.2
with h5py.File(p, "r") as f:
    for k in sorted(f.keys(), key=int):
        y = f[k]["lc_flux"][:]
        tic = f[k]["tic_metadata"][:]
        med = float(np.median(y))
        depth = med - float(y.min())
        rad = float(tic[3])
        rp_re = (max(depth, 0.0) ** 0.5) * rad * RSUN_REARTH
        print(
            k,
            int(f[k].attrs["label"]),
            round(depth * 1e6, 1),
            round(float(tic[0]), 2),
            round(float(tic[1]), 1),
            round(rad, 3),
            round(rp_re, 2),
        )
