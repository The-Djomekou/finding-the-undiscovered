import numpy as np
import pandas as pd

from src import scoring

FEATURES = sorted({f for cols in scoring.CRITERIA.values() for f in cols})


def make_tables():
    rng = np.random.default_rng(0)
    n = 400
    earth = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    earth["site_id"] = [f"E{i}" for i in range(n)]
    earth["x"] = (np.arange(n) % 20) * 5000.0
    earth["y"] = (np.arange(n) // 20) * 5000.0
    earth["lat"] = 46.0 + earth["y"] / 111_000
    earth["lon"] = -72.0 + earth["x"] / 77_000
    # Target copies Earth square E7 exactly, so E7 must score 1.0 against it.
    target = earth.loc[[7], FEATURES].copy()
    for f in scoring.RELATIVE:
        target[f] = earth[f].rank(pct=True).iloc[7]
    target["site_id"], target["body"] = "MARS_TEST", "Mars"
    return earth, target.reset_index(drop=True)


def test_identical_square_scores_one():
    earth, target = make_tables()
    res = scoring.best_match(earth, target)
    assert res.loc[7, "S_mars"] > 0.999
    assert res.loc[7, "tier_mars"] == "Tier 1"
    assert res["S_mars"].between(0, 1).all()


def test_bands_and_tiers():
    assert scoring.band(0.9) == "Strong match"
    assert scoring.band(0.7) == "Match"
    assert scoring.band(0.6) == "Near miss"
    assert scoring.band(0.1) == "Mismatch"
    assert scoring.tier(1 / 6) == "Tier 4"
    assert scoring.tier(2 / 6) == "Tier 3"
    assert scoring.tier(3 / 6) == "Tier 2"
    assert scoring.tier(5 / 6) == "Tier 1"
    assert scoring.tier(0) is None


def test_one_spread_off_gives_085():
    # Terrain has 3 features; one off by 1 z-unit -> exp(-(1/3)/2) = 0.846
    assert abs(np.exp(-(1 / 3) / 2) - 0.846) < 0.001


def test_exclusion_and_grouping():
    earth, _ = make_tables()
    known = pd.DataFrame({"lat": [46.0], "lon": [-72.0]})
    keep = scoring.exclude_known(earth, known, buffer_km=50)
    assert not keep[0] and keep.sum() < len(earth)
    qualifies = np.zeros(len(earth), dtype=bool)
    qualifies[[0, 1, 21, 300]] = True  # 0, 1, 21 touch; 300 is alone
    sites = scoring.group_into_sites(earth, qualifies)
    assert sites[0] == sites[1] == sites[21] != 0
    assert sites[300] not in (0, sites[0])
