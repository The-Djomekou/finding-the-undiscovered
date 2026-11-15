"""Terrain measures at 120 m, the same for an Earth site and the lunar fingerprint.

The Artemis III SDT report gives LOLA LDEM GDR at about 118 m/px, so every slope is computed
at 120 m: comparing a 30 m slope with a 118 m slope would bias the comparison.
"""
import numpy as np
from scipy import ndimage

SLOPE_PASSABLE_DEG = 15.0  # team choice: passable terrain is 0 to 15 deg (the SDT report sets none)


def slope_deg(dem, cell_m):
    gy, gx = np.gradient(dem, cell_m)
    return np.degrees(np.arctan(np.hypot(gx, gy)))


def roughness_std3(dem):
    """Standard deviation of elevation in a 3x3 window."""
    m = ndimage.uniform_filter(dem, 3)
    m2 = ndimage.uniform_filter(dem * dem, 3)
    return np.sqrt(np.clip(m2 - m * m, 0, None))


def block_stats(dem, cell_m, block_m=10000, sub_m=1000):
    """Terrain statistics per block, computed like the Earth Engine export (two stages).

    Stage 1: slope median and IQR per 1 km sub-block. Stage 2: their mean over the block.
    Roughness is the mean 3x3 standard deviation; relief is max minus min elevation.
    """
    s = slope_deg(dem, cell_m)
    r = roughness_std3(dem)
    n = max(int(block_m // cell_m), 1)
    m = max(int(round(sub_m / cell_m)), 1)
    rows = []
    for i in range(0, dem.shape[0] - n + 1, n):
        for j in range(0, dem.shape[1] - n + 1, n):
            blk = (slice(i, i + n), slice(j, j + n))
            z, sl, ro = dem[blk], s[blk], r[blk]
            if np.isfinite(z).mean() < 0.8:
                continue
            med, iqr = [], []
            for a in range(0, n - m + 1, m):
                for b in range(0, n - m + 1, m):
                    sub = sl[a:a + m, b:b + m]
                    if np.isfinite(sub).any():
                        p25, p50, p75 = np.nanpercentile(sub, [25, 50, 75])
                        med.append(p50)
                        iqr.append(p75 - p25)
            rows.append({"slope_med_deg": float(np.mean(med)), "slope_iqr_deg": float(np.mean(iqr)),
                         "roughness_m": float(np.nanmean(ro)), "relief_m": float(np.nanmax(z) - np.nanmin(z))})
    return rows


def traffic_at_centre(dem, cell_m, radii_m=(2000, 10000, 50000), passable_deg=SLOPE_PASSABLE_DEG):
    """Share of passable terrain within each radius of the DEM centre."""
    s = slope_deg(dem, cell_m)
    ny, nx = dem.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    d = np.hypot((yy - ny / 2) * cell_m, (xx - nx / 2) * cell_m)
    ok = np.isfinite(s)
    out = {}
    for r in radii_m:
        m = (d <= r) & ok
        out[f"traffic_{r // 1000}km"] = float(np.mean(s[m] < passable_deg)) if m.any() else np.nan
        out[f"traffic_{r // 1000}km_complete"] = bool(min(ny, nx) * cell_m / 2 >= r)
    return out
