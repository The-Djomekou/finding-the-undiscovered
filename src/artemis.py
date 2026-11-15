# Written with assistance from Claude (Anthropic); reviewed by the team.
"""Moon base profile: score Earth cells against the Artemis III criteria transposed to Earth.

Each criterion score is 0 to 1 and is met at 0.70 (same bands and tiers as src/scoring.py).
Criteria scored globally for the Moon profile (numbering of the 8 analog criteria):
  s1 surface material  sand_pct / 100                      (partial proxy for grain size)
  s2 terrain           similarity of slope median, slope IQR, roughness and relief to the
                       Shackleton fingerprint (docs/moon_fingerprint.csv), exp(-d^2/2)
  s3 illumination      low_sun_days / Earth maximum        (low solar angles, SDT 4.1)
  s4 ice               ice_class: surficial 1.0, shallow 0.75, deep 0.5, dry 0 (SDT 5.2)
  s7 subsurface        regolith_depth_cm / 250              (2.5 m mechanised coring, SDT 6.1)
  s8 mobility          mean of traffic_2km, traffic_10km, traffic_50km (xEVA, LTV, pressurised rover)
Overall score S: AHP weights on four groups, illumination 0.467 > ice 0.277 > terrain and
mobility 0.160 > regolith and soil 0.095 (pairwise matrix below; CR < 0.1).
Binary constraints (facts outside the report): no vegetation (NDVI max < 0.2), and some flat
ground to land or build on (traffic_2km >= 0.05).
"""
import numpy as np
import pandas as pd

from src import scoring
from src.illumination import dark_days, low_sun_days, max_low_sun_days

GROUPS = ["illumination", "ice", "terrain_mobility", "regolith_soil"]
# Saaty pairwise comparisons, rows and columns in GROUPS order
PAIRWISE = np.array([[1, 2, 3, 4],
                     [1 / 2, 1, 2, 3],
                     [1 / 3, 1 / 2, 1, 2],
                     [1 / 4, 1 / 3, 1 / 2, 1]], dtype=float)
TERRAIN_FEATURES = ["slope_med_deg", "slope_iqr_deg", "roughness_m", "relief_m"]
ICE_SCORE = {3: 1.0, 2: 0.75, 1: 0.5, 0: 0.0}
ICE_LABEL = {3: "Surficial", 2: "Shallow (<= 50 cm)", 1: "Deep (50-100 cm)", 0: "Dry"}
NDVI_MAX = 0.2
GROUND_MIN_SHARE = 0.5  # at least half of a cell must be ice-free land (neither permanent ice nor water)
FLAT_MIN = 0.05
REGOLITH_TARGET_CM = 250.0
CRITERION_COLS = ["s1", "s2", "s3", "s4", "s7", "s8"]
CRITERION_NAMES = {"s1": "Surface material", "s2": "Terrain", "s3": "Illumination",
                   "s4": "Water, ice", "s7": "Subsurface", "s8": "Mobility"}


def ahp_weights(matrix=PAIRWISE):
    """Principal-eigenvector weights and consistency ratio of a pairwise matrix."""
    vals, vecs = np.linalg.eig(matrix)
    k = np.argmax(vals.real)
    w = np.abs(vecs[:, k].real)
    w = w / w.sum()
    n = matrix.shape[0]
    ci = (vals[k].real - n) / (n - 1)
    ri = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}[n]
    return dict(zip(GROUPS, w)), (ci / ri if ri else 0.0)


def terrain_similarity(df, fingerprint):
    """s2: similarity of each cell's terrain to the lunar fingerprint, 0 to 1."""
    if fingerprint is None:
        return pd.Series(np.nan, index=df.index)
    diffs = []
    for f in TERRAIN_FEATURES:
        if f not in df or f not in fingerprint:
            continue
        e = df[f].astype(float).to_numpy()
        med, scale = scoring._robust_scale(e)
        diffs.append(((e - med) / scale - (float(fingerprint[f]) - med) / scale) ** 2)
    if not diffs:
        return pd.Series(np.nan, index=df.index)
    with np.errstate(invalid="ignore"):
        d2 = np.nanmean(np.vstack(diffs), axis=0)
    return pd.Series(np.exp(-d2 / 2), index=df.index)


def criterion_scores(df, fingerprint=None):
    out = pd.DataFrame(index=df.index)
    out["low_sun_days"] = low_sun_days(df["lat"].to_numpy())
    out["dark_days"] = dark_days(df["lat"].to_numpy())
    out["s1"] = (df["sand_pct"] / 100).clip(0, 1) if "sand_pct" in df else np.nan
    out["s2"] = terrain_similarity(df, fingerprint)
    out["s3"] = (out["low_sun_days"] / max_low_sun_days()).clip(0, 1)
    out["s4"] = df["ice_class"].round().map(ICE_SCORE) if "ice_class" in df else np.nan
    out["s7"] = ((df["regolith_depth_cm"] / REGOLITH_TARGET_CM).clip(0, 1)
                 if "regolith_depth_cm" in df else np.nan)
    traffic = [c for c in ["traffic_2km", "traffic_10km", "traffic_50km"] if c in df]
    out["s8"] = df[traffic].mean(axis=1) if traffic else np.nan
    out["s5"] = np.nan  # wind and dust: not applicable on the Moon
    return out


def score(df, fingerprint=None, weights=None):
    """Criterion scores, AHP overall score, counts, match ratio, tier and constraint flags."""
    sc = criterion_scores(df, fingerprint)
    w = weights or ahp_weights()[0]
    groups = pd.DataFrame({
        "illumination": sc["s3"],
        "ice": sc["s4"],
        "terrain_mobility": sc[["s2", "s8"]].mean(axis=1),
        "regolith_soil": sc[["s7", "s1"]].mean(axis=1),
    })
    wv = np.array([w[g] for g in GROUPS])
    g = groups[GROUPS].to_numpy(dtype=float)
    ok = np.isfinite(g)
    with np.errstate(invalid="ignore", divide="ignore"):
        sc["S"] = np.nansum(g * wv, axis=1) / (ok * wv).sum(axis=1)
    crit = sc[CRITERION_COLS].to_numpy(dtype=float)
    scored = np.isfinite(crit)
    sc["n_scored"] = scored.sum(axis=1)
    sc["n_met"] = (np.nan_to_num(crit, nan=0) >= scoring.MET).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sc["match_ratio"] = sc["n_met"] / sc["n_scored"]
    sc["tier"] = sc["match_ratio"].map(scoring.tier)
    idx = df.index
    veg = df["ndvi_max"] < NDVI_MAX if "ndvi_max" in df else pd.Series(True, index=idx)
    flat = df["traffic_2km"] >= FLAT_MIN if "traffic_2km" in df else pd.Series(True, index=idx)
    ground = pd.Series(True, index=idx)
    if "ice_frac" in df:
        # a landing analog needs ground: at least half of the cell must be neither ice nor water
        # (testing ice and water separately let half-ice, half-sea coast cells through)
        known = df["ice_frac"].notna() & df["water_frac"].notna()
        land = (1 - df["ice_frac"] - df["water_frac"]) >= GROUND_MIN_SHARE
        ground = ~known | land
        # MODIS gives no vegetation value over polar ice; ice-free ground without one is bare
        if "ndvi_max" in df:
            veg = veg | (df["ndvi_max"].isna() & known & land)
    sc["veg_ok"], sc["flat_ok"], sc["ground_ok"] = veg, flat, ground
    sc["passes_constraints"] = veg & flat & ground
    return sc
