# Written with assistance from Claude (Anthropic); reviewed by the team.
"""Summary of the Moon results for the write-up, and a check that no site touches a known analog.

    python scripts/tools/summarize_sites.py

Prints: the 20 best sites, Tier 2 sites per country, the best 2 sites per continent, and for every
known analog (docs/known_analogs.csv, used by the scoring) and every extra analog
(docs/known_analogs_extra.csv, approximate positions; both lists are excluded by 03) the distance to the
nearest cell that belongs to a site.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
BUFFER_KM = 50


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def main():
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    sites = pd.read_parquet(PROC / "sites.parquet")
    cells = pd.read_parquet(PROC / "earth_scores_moon.parquet", columns=["lat", "lon", "site_label"])
    cells = cells[cells["site_label"] > 0]
    label = cells.merge(sites[["lat", "lon", "site_code"]], on=["lat", "lon"], how="inner")
    code_of = dict(zip(label["site_label"], label["site_code"]))
    cells["site_code"] = cells["site_label"].map(code_of)

    cols = [c for c in ["rank", "site", "lat", "lon", "tier", "S", "n_met", "s1", "s3", "s4", "s8",
                        "low_sun_days", "ice_label", "sand_pct", "ndvi_max", "ice_frac", "water_frac",
                        "slope_med_deg", "relief_m", "area_km2"] if c in sites]
    print(f"## 20 best of {len(sites):,} sites\n")
    print(sites.head(20)[cols].round(2).to_string(index=False))

    t2 = sites[sites["tier"] == "Tier 2"]
    print(f"\n## Tier 2 sites per country ({len(t2)})\n")
    print(t2.groupby(["continent", "country"]).agg(sites=("site_code", "size"), best_S=("S", "max"),
                                                    lat_min=("lat", "min"), lat_max=("lat", "max"))
          .round(2).sort_values("sites", ascending=False).to_string())

    print("\n## Best sites on each continent (all sites in the highest tier reached there)\n")
    if "best_on_continent" in sites:
        bc = sites[sites["best_on_continent"]]
        print(bc.groupby("continent").agg(best_tier=("tier", "first"), sites=("site_code", "size"),
                                          best_S=("S", "max")).to_string())
        print("\nTop 10 of each (the full list is docs/best_sites_by_continent.csv):\n")
        print(bc.sort_values(["continent", "continent_rank"]).groupby("continent").head(10)[cols]
              .round(2).to_string(index=False))

    print(f"\n## Known analogs: nearest site cell (a site cell closer than {BUFFER_KM} km = problem)\n")
    rows = []
    for fname, status in (("known_analogs.csv", "main list"), ("known_analogs_extra.csv", "extra list")):
        path = ROOT / "docs" / fname
        if not path.exists():
            continue
        for _, a in pd.read_csv(path).iterrows():
            d = haversine_km(a["lat"], a["lon"], cells["lat"].to_numpy(), cells["lon"].to_numpy())
            i = int(np.argmin(d))
            rows.append({"analog": a["name"], "list": status, "nearest_site_km": round(float(d[i]), 1),
                         "nearest_site": cells["site_code"].iloc[i],
                         "inside_50km": "YES" if d[i] < BUFFER_KM else "no"})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
