"""Similarity scoring for Finding the Undiscovered.

Formulas, as in the project brief:
    d_c = sqrt(mean_i (z_i_earth - z_i_target) ** 2)   distance on criterion c
    s_c = exp(-d_c ** 2 / 2)                          criterion score, 0 to 1
    S   = sum_c w_c * s_c / sum_c w_c                 overall score, 0 to 1
A criterion is met when s_c >= 0.70. Match ratio = criteria met / criteria scored.
"""
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.stats import norm

# Criteria scored globally, and the feature columns each one uses.
# Criteria 6 and 7 are assessed later, for dossier sites only.
CRITERIA = {
    1: ["albedo", "iron_proxy"],
    2: ["slope_deg", "roughness_tri", "relief_m"],
    3: ["tmin_k", "tday_range_k", "tyear_range_k", "sun_elev_deg"],
    4: ["ice_flag", "hydrogen_or_soilmoist", "hydrated_flag"],
    5: ["dust_aod", "wind_ms", "aeolian_flag"],
    8: ["gentle_slope_frac", "rock_abundance"],
}
NOT_APPLICABLE = {"Moon": {5}, "Mars": set()}

# "relative": Earth cannot reach the target's absolute value, so compare
# percentiles. Target rows hold the percentile (0 to 1) of the target within
# its own body; Earth values are converted to percentiles within Earth.
RELATIVE = {"tmin_k", "tday_range_k", "tyear_range_k", "sun_elev_deg",
            "hydrogen_or_soilmoist", "dust_aod", "wind_ms", "iron_proxy"}

MET = 0.70
BANDS = [(0.85, "Strong match"), (0.70, "Match"), (0.50, "Near miss"), (0.0, "Mismatch")]
TIERS = [(0.75, "Tier 1"), (0.50, "Tier 2"), (0.25, "Tier 3"), (1e-9, "Tier 4")]


def band(score):
    """Band name for a criterion score."""
    if pd.isna(score):
        return None
    return next(name for cut, name in BANDS if score >= cut)


def tier(match_ratio):
    """Tier name for a match ratio; None if no criterion is met."""
    if pd.isna(match_ratio) or match_ratio <= 0:
        return None
    return next(name for cut, name in TIERS if match_ratio >= cut)


def _robust_scale(values):
    med = np.nanmedian(values)
    scale = 1.4826 * np.nanmedian(np.abs(values - med))
    if not scale:
        scale = np.nanstd(values) or 1.0
    return med, scale


def _to_z(p):
    return norm.ppf(np.clip(p, 0.001, 0.999))


def normalise(earth, targets):
    """Return z-scored copies of the Earth and target feature tables."""
    features = sorted({f for cols in CRITERIA.values() for f in cols})
    ze, zt = pd.DataFrame(index=earth.index), pd.DataFrame(index=targets.index)
    for f in features:
        e = earth[f].astype(float).to_numpy()
        t = targets[f].astype(float).to_numpy()
        if f in RELATIVE:
            ze[f] = _to_z(pd.Series(e).rank(pct=True).to_numpy())
            zt[f] = _to_z(t)
        else:
            med, scale = _robust_scale(e)
            ze[f] = (e - med) / scale
            zt[f] = (t - med) / scale
    return ze, zt


def score_against_target(ze, zt_row, body, weights=None):
    """Criterion scores, overall score and counts for every Earth square vs one target."""
    weights = weights or {c: 1.0 for c in CRITERIA}
    out = pd.DataFrame(index=ze.index)
    num = np.zeros(len(ze))
    den = np.zeros(len(ze))
    met = np.zeros(len(ze))
    scored = np.zeros(len(ze))
    for c, cols in CRITERIA.items():
        if c in NOT_APPLICABLE.get(body, set()):
            continue
        diff = ze[cols].to_numpy() - zt_row[cols].to_numpy(dtype=float)
        with np.errstate(invalid="ignore"):
            d2 = np.nanmean(diff ** 2, axis=1)
        s = np.exp(-d2 / 2)
        out[f"s{c}"] = s
        ok = ~np.isnan(s)
        num[ok] += weights[c] * s[ok]
        den[ok] += weights[c]
        met[ok] += s[ok] >= MET
        scored[ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        out["S"] = num / den
        out["match_ratio"] = met / scored
    out["n_met"], out["n_scored"] = met, scored
    return out


def best_match(earth, targets, weights=None):
    """For each Earth square, keep the best-scoring target on each body."""
    ze, zt = normalise(earth, targets)
    result = earth[["site_id"]].copy()
    for body in ["Moon", "Mars"]:
        best = None
        for idx, row in targets[targets["body"] == body].iterrows():
            sc = score_against_target(ze, zt.loc[idx], body, weights)
            sc["target"] = row["site_id"]
            if best is None:
                best = sc
            else:
                better = sc["S"] > best["S"]
                best.loc[better] = sc.loc[better]
        if best is None:
            continue
        best["tier"] = best["match_ratio"].map(tier)
        result = result.join(best.add_suffix(f"_{body.lower()}"))
    return result


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = (np.sin((lat2 - lat1) / 2) ** 2
         + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def exclude_known(earth, known, buffer_km=50):
    """Boolean mask: True for squares farther than buffer_km from every known analog."""
    keep = np.ones(len(earth), dtype=bool)
    for _, k in known.iterrows():
        d = haversine_km(earth["lat"].to_numpy(), earth["lon"].to_numpy(), k["lat"], k["lon"])
        keep &= d > buffer_km
    return keep


def group_into_sites(squares, qualifies, x_col="x", y_col="y", grid_m=5000):
    """Label touching qualifying squares (8 neighbours) as one site; 0 = not in a site."""
    ix = np.round((squares[x_col] - squares[x_col].min()) / grid_m).astype(int).to_numpy()
    iy = np.round((squares[y_col] - squares[y_col].min()) / grid_m).astype(int).to_numpy()
    grid = np.zeros((iy.max() + 1, ix.max() + 1), dtype=bool)
    grid[iy[qualifies], ix[qualifies]] = True
    labels, _ = ndimage.label(grid, structure=np.ones((3, 3)))
    return labels[iy, ix] * qualifies
