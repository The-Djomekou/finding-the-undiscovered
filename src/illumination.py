"""Sunlight on Earth, transposed from the Artemis III illumination criteria.

Global (from latitude only, no data needed):
    low_sun_days  days per year when the sun rises but its highest elevation stays under 10 deg
    dark_days     days per year of polar night (the sun never rises)
Site level (needs a DEM, see horizon_angles and lighting_metrics):
    lit_frac      share of all hours of the year a cell is in direct sunlight
    lit_when_up   share of the hours when the sun is up that a cell is actually lit (cast shadows)
    lit_best      lit_frac of the best-lit cell (control: must stay below 1 on Earth)
    illum_contrast  lit_frac spread between well-lit rims and shadowed floors (p95 - p05)
    psr_frac      share of cells never lit during the year
"""
import numpy as np

LOW_SUN_DEG = 10.0


def solar_declination(doy):
    """Solar declination in degrees for day of year (Spencer 1971)."""
    g = 2 * np.pi * (np.asarray(doy) - 1) / 365.0
    d = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
         + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    return np.degrees(d)


_DEC = solar_declination(np.arange(1, 366))


def _per_latitude(lat, low=LOW_SUN_DEG, horizon=0.0):
    lat = np.asarray(lat, dtype=float)
    keys = np.round(lat, 2)
    uniq, inv = np.unique(keys, return_inverse=True)
    noon = 90.0 - np.abs(uniq[:, None] - _DEC[None, :])  # highest sun elevation of each day
    low_sun = ((noon > horizon) & (noon < low)).sum(axis=1)
    dark = (noon <= horizon).sum(axis=1)
    return low_sun[inv].reshape(lat.shape), dark[inv].reshape(lat.shape)


def low_sun_days(lat, low=LOW_SUN_DEG):
    return _per_latitude(lat, low)[0]


def dark_days(lat):
    return _per_latitude(lat)[1]


def max_low_sun_days(low=LOW_SUN_DEG):
    """Largest low_sun_days anywhere on Earth (used to scale the illumination score)."""
    return int(low_sun_days(np.arange(0, 90.01, 0.05), low).max())


# ---------------------------------------------------------------- site level, needs a DEM
def horizon_angles(dem, cell_m, n_az=36, max_dist_m=15000):
    """Horizon elevation angle (deg) in n_az azimuths for every cell of a DEM on a metric grid.

    Azimuth 0 = grid north (up the array), clockwise. Earth curvature is included.
    Returns an array of shape (n_az, ny, nx).
    """
    ny, nx = dem.shape
    z = np.where(np.isfinite(dem), dem, np.nanmin(dem)).astype(np.float32)
    out = np.full((n_az, ny, nx), -90.0, dtype=np.float32)
    steps = np.unique(np.round(np.geomspace(cell_m, max_dist_m, 60) / cell_m).astype(int))
    earth_r = 6_371_000.0
    for k in range(n_az):
        az = np.radians(360.0 * k / n_az)
        best = np.full((ny, nx), -np.inf, dtype=np.float32)
        for s in steps:
            di, dj = -int(round(s * np.cos(az))), int(round(s * np.sin(az)))
            if abs(di) >= ny or abs(dj) >= nx:
                continue
            dist = np.hypot(di, dj) * cell_m
            shifted = np.full((ny, nx), np.nan, dtype=np.float32)
            src_i = slice(max(di, 0), ny + min(di, 0))
            dst_i = slice(max(-di, 0), ny + min(-di, 0))
            src_j = slice(max(dj, 0), nx + min(dj, 0))
            dst_j = slice(max(-dj, 0), nx + min(-dj, 0))
            shifted[dst_i, dst_j] = z[src_i, src_j]
            drop = dist ** 2 / (2 * earth_r)
            ang = np.degrees(np.arctan2(shifted - z - drop, dist))
            best = np.fmax(best, ang)
        out[k] = np.where(np.isfinite(best), best, 0.0)
    return out


def sun_track(lat, day_step=5, hour_step=1.0):
    """Sun elevation and azimuth (deg) sampled over a year at a latitude (local solar time)."""
    doy = np.arange(1, 366, day_step)
    hours = np.arange(0, 24, hour_step) + hour_step / 2
    dec = np.radians(solar_declination(doy))[:, None]
    ha = np.radians(15.0 * (hours - 12.0))[None, :]
    phi = np.radians(lat)
    sin_el = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(ha)
    el = np.degrees(np.arcsin(np.clip(sin_el, -1, 1)))
    az = np.degrees(np.arctan2(np.sin(ha) * np.cos(dec),
                               np.cos(ha) * np.sin(phi) * np.cos(dec) - np.sin(dec) * np.cos(phi)))
    az = (az + 180.0) % 360.0  # 0 = north, clockwise
    return el.ravel(), az.ravel()


def lighting_metrics(dem, cell_m, lat, n_az=36, max_dist_m=15000):
    """Per-cell lit fractions and site summary metrics for a DEM centred near latitude lat."""
    hor = horizon_angles(dem, cell_m, n_az, max_dist_m)
    el, az = sun_track(lat)
    total = el.size
    up = el > 0
    lit_hours = np.zeros(dem.shape, dtype=np.float32)
    bins = np.round(az / (360.0 / n_az)).astype(int) % n_az
    for k in range(n_az):
        sel = up & (bins == k)
        if sel.any():
            lit_hours += (el[sel][:, None, None] > hor[k][None]).sum(axis=0)
    lit_frac = lit_hours / total
    lit_when_up = lit_hours / max(up.sum(), 1)
    valid = np.isfinite(dem)
    lf = lit_frac[valid]
    return {
        "lit_frac": lit_frac,
        "lit_when_up_map": lit_when_up,
        "lit_when_up": float(np.mean(lit_when_up[valid])),
        "lit_best": float(lf.max()),
        "illum_contrast": float(np.percentile(lf, 95) - np.percentile(lf, 5)),
        "psr_frac": float(np.mean(lf == 0)),
        "sun_up_share": float(up.mean()),
    }
