"""Quick look at an Earth Engine export: one map per band, plus min / median / max.

    python scripts/tools/preview_tif.py data/raw/earth_test/earth_features_10km_test_quebec.tif

Saves a PNG next to the GeoTIFF and prints a short table. For checking the data only.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import rasterio  # noqa: E402

BAND_NAMES = ["slope_med_deg", "slope_iqr_deg", "roughness_m", "relief_m", "traffic_2km",
              "traffic_10km", "traffic_50km", "ice_class", "soil_t1max_k", "soil_t2max_k",
              "soil_t3max_k", "sand_pct", "ndvi_max", "aridity"]


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    path = Path(sys.argv[1])
    with rasterio.open(path) as src:
        # no-data cells come out of Earth Engine as NaN: mask them so the statistics ignore them
        data = np.ma.masked_invalid(src.read(masked=True).astype("float64"))
        names = [d or BAND_NAMES[i] for i, d in enumerate(src.descriptions)] \
            if len(src.descriptions) == len(BAND_NAMES) else BAND_NAMES[:src.count]
        b = src.bounds
    extent = [b.left, b.right, b.bottom, b.top]
    ncol = 4
    nrow = int(np.ceil(len(names) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3.4 * nrow), constrained_layout=True)
    print(f"{'band':15s} {'valid %':>8s} {'min':>9s} {'median':>9s} {'max':>9s}")
    for ax, name, band in zip(axes.flat, names, data):
        vals = band.compressed()
        if vals.size:
            lo, hi = np.percentile(vals, [2, 98])
            print(f"{name:15s} {100 * vals.size / band.size:8.1f} {vals.min():9.3f} "
                  f"{np.median(vals):9.3f} {vals.max():9.3f}")
        else:
            lo, hi = 0, 1
            print(f"{name:15s} {'0.0':>8s}  (empty)")
        cmap = "Blues" if name == "ice_class" else "viridis"
        im = ax.imshow(band, extent=extent, cmap=cmap, vmin=lo, vmax=hi if hi > lo else lo + 1)
        ax.set_title(name, fontsize=10)
        ax.tick_params(labelsize=7)
        fig.colorbar(im, ax=ax, shrink=0.8)
    for ax in list(axes.flat)[len(names):]:
        ax.axis("off")
    out = path.with_suffix(".png")
    fig.suptitle(path.name, fontsize=11)
    fig.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
