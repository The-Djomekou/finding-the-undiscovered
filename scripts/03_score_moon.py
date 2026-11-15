# Written with assistance from Claude (Anthropic); reviewed by the team.
"""Block 2: score every Earth cell against the Artemis III Moon base profile and group sites.

    python scripts/03_score_moon.py
Inputs : data/processed/earth_features.parquet, docs/moon_fingerprint.csv (optional),
         docs/known_analogs.csv
Outputs: data/processed/earth_scores_moon.parquet  every cell, all scores (large, not in git)
         data/processed/sites.parquet               one row per site, read by the app and the book
         data/processed/sites_moon.csv              the same, easy to open in Numbers or Excel
         docs/results_summary.md                    sites per tier and continent
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import artemis, scoring  # noqa: E402

PROC = ROOT / "data" / "processed"
TIER_ORDER = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3, "Tier 4": 4}
SITE_MAX_M = 50_000  # a site is at most 50 km across: the pressurised-rover range (SDT 4.2)


def sites_by_tier(cells, qualifies, grid_m):
    """Touching qualifying cells of the same tier form a site, cut into blocks of 50 km."""
    site = np.zeros(len(cells), dtype=np.int64)
    step = SITE_MAX_M / 111_320.0                      # 50 km in degrees of latitude
    lat, lon = cells["lat"].to_numpy(), cells["lon"].to_numpy()
    by = np.floor(lat / step).astype(np.int64)
    width = step / np.maximum(np.cos(np.radians((by + 0.5) * step)), 0.05)
    bx = np.floor((lon + 180.0) / width).astype(np.int64)
    next_id = 1
    for t in TIER_ORDER:
        m = qualifies & (cells["tier"] == t).to_numpy()
        if not m.any():
            continue
        lab = scoring.group_into_sites(cells, m, grid_m=grid_m)
        keys = pd.Series(list(zip(lab[m], bx[m], by[m])))
        ids, _ = pd.factorize(keys)
        site[m] = ids + next_id
        next_id = site.max() + 1
    return site


def load_fingerprint():
    p = ROOT / "docs" / "moon_fingerprint.csv"
    if not p.exists():
        print("no docs/moon_fingerprint.csv: criterion 2 (terrain) skipped; run 00_moon_fingerprint.py")
        return None
    fp = pd.read_csv(p)
    return dict(zip(fp["feature"], fp["value"]))


def main():
    df = pd.read_parquet(PROC / "earth_features.parquet")
    known = scoring.load_known(ROOT / "docs")  # main list plus the extra list of training sites
    print(f"excluding 50 km around {len(known)} known analogs")
    sc = artemis.score(df, load_fingerprint())
    cells = pd.concat([df, sc], axis=1)
    cells["outside_known_buffer"] = scoring.exclude_known(cells, known, buffer_km=50)
    cells["qualifies"] = (cells["n_met"] >= 1) & cells["passes_constraints"] & cells["outside_known_buffer"]
    grid_m = float(cells["grid_m"].iloc[0])
    cells["site_label"] = sites_by_tier(cells, cells["qualifies"].to_numpy(), grid_m)
    cells.to_parquet(PROC / "earth_scores_moon.parquet", index=False)

    q = cells[cells["site_label"] > 0].copy()
    q["tier_rank"] = q["tier"].map(TIER_ORDER)
    q = q.sort_values(["tier_rank", "S"], ascending=[True, False])
    best = q.groupby("site_label").head(1).copy()  # best cell of each site represents it
    size = q.groupby("site_label").size()
    area = q.groupby("site_label")["cell_area_km2"].sum()
    best["n_cells"] = best["site_label"].map(size).to_numpy()
    best["area_km2"] = best["site_label"].map(area).to_numpy()
    best = best.sort_values(["tier_rank", "S"], ascending=[True, False]).reset_index(drop=True)
    best["rank"] = np.arange(1, len(best) + 1)
    best["site_code"] = [f"MOON-{r:04d}" for r in best["rank"]]
    best["site"] = [f"{c} · {k}" for c, k in zip(best["site_code"], best["country"].astype(str))]
    # rank inside each continent; a continent's best sites are all its sites in the highest tier
    # reached on that continent (no fixed number)
    best["continent_rank"] = best.groupby("continent").cumcount() + 1
    top_tier = best.groupby("continent")["tier_rank"].transform("min")
    best["best_on_continent"] = best["tier_rank"] == top_tier
    best["body"] = "Moon"
    best["target"] = "Artemis III south pole profile"
    best["ice_label"] = best["ice_class"].round().map(artemis.ICE_LABEL)
    keep = ["rank", "continent_rank", "best_on_continent", "site_code", "site", "lat", "lon", "country",
            "continent", "body", "target",
            "tier", "S", "match_ratio", "n_met", "n_scored", "s1", "s2", "s3", "s4", "s5", "s7", "s8",
            "low_sun_days", "dark_days", "ice_class", "ice_label", "traffic_2km", "traffic_10km",
            "traffic_50km", "sand_pct", "regolith_depth_cm", "slope_med_deg", "slope_iqr_deg",
            "roughness_m", "relief_m", "ndvi_max", "aridity", "ice_frac", "water_frac", "n_cells",
            "area_km2", "synthetic"]
    sites = best[[c for c in keep if c in best]]
    sites.to_parquet(PROC / "sites.parquet", index=False)
    sites.to_csv(PROC / "sites_moon.csv", index=False)
    best_cont = sites[sites["best_on_continent"]].sort_values(["continent", "continent_rank"])
    best_cont.to_csv(ROOT / "docs" / "best_sites_by_continent.csv", index=False)

    table = pd.crosstab(sites["continent"], sites["tier"]).reindex(columns=list(TIER_ORDER), fill_value=0)
    table["All"] = table.sum(axis=1)
    lines = ["# Results summary: Moon base profile", "",
             f"Cells scored: {len(cells):,}. Cells qualifying (at least 1 criterion met, constraints "
             f"passed, outside 50 km of a known analog): {int(cells['qualifies'].sum()):,}. "
             f"Sites (touching cells of the same tier, at most 50 km across): {len(sites):,}.", "",
             "Why cells drop out (a cell can fail several):", "",
             f"- vegetation (annual max NDVI at or above {artemis.NDVI_MAX}, or no NDVI value "
             f"on ice-covered or unknown ground): {int((~cells['veg_ok']).sum()):,}",
             f"- too steep (less than {artemis.FLAT_MIN:.0%} passable ground within 2 km): "
             f"{int((~cells['flat_ok']).sum()):,}",
             f"- not ground (less than half of the cell is ice-free land, MODIS land cover): "
             f"{int((~cells['ground_ok']).sum()):,}",
             f"- no criterion met: {int((cells['n_met'] < 1).sum()):,}",
             f"- within 50 km of a known analog: {int((~cells['outside_known_buffer']).sum()):,}", ""]
    if bool(cells["synthetic"].any()):
        lines.insert(2, "**SYNTHETIC TEST DATA: these numbers test the pipeline and are not results.**\n")
    lines += ["| Continent | " + " | ".join(table.columns) + " |",
              "|---|" + "---|" * len(table.columns)]
    lines += [f"| {c} | " + " | ".join(str(v) for v in row) + " |" for c, row in table.iterrows()]
    if sites.empty:
        lines += ["", "No site in this region: every cell failed at least one rule above."]
    else:
        lines += ["", "Best sites on each continent (every site in the highest tier reached there; "
                  "full list in docs/best_sites_by_continent.csv):", "",
                  "| Continent | Best tier | Sites in it | Best site | S |", "|---|---|---|---|---|"]
        for cont, g in best_cont.groupby("continent"):
            top = g.iloc[0]
            lines.append(f"| {cont} | {top['tier']} | {len(g):,} | {top['site']} | {top['S']:.2f} |")
    (ROOT / "docs" / "results_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
