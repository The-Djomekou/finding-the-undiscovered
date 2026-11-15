# Written with assistance from Claude (Anthropic); reviewed by the team.
"""Block 3: dossier measurements for the top sites from the Copernicus DEM at 120 m.

For each selected site, reads Copernicus GLO-90 tiles straight from the public AWS bucket
(no login), resamples them to 120 m on a grid centred on the site, and computes:
  lit_when_up, lit_best, illum_contrast, psr_frac   cast shadows over a full year
  traffic_2km, traffic_10km, traffic_50km           passable share (slope < 15 deg) around the centre
  slope, roughness and relief of the central 10 km block
Control from the criteria document: lit_best must stay below 1 on Earth.

    python scripts/05_site_detail.py --top 6             # best 6 sites from Tiers 1 and 2
    python scripts/05_site_detail.py --top 6 --per-continent 2   # plus the best 2 of each continent
    python scripts/05_site_detail.py --sites MOON-0003,MOON-0010
Outputs per site: docs/dossiers/<code>/metrics.json, lighting.png, slope.png
"""
import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from rasterio.transform import from_origin  # noqa: E402
from rasterio.warp import Resampling, reproject  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.illumination import lighting_metrics  # noqa: E402
from src.terrain import SLOPE_PASSABLE_DEG, block_stats, slope_deg, traffic_at_centre  # noqa: E402

CELL = 120.0
BUCKET = "https://copernicus-dem-90m.s3.amazonaws.com"


def tile_urls(lat, lon, half_km):
    dlat = half_km / 111.0
    dlon = min(half_km / (111.0 * max(math.cos(math.radians(lat)), 0.05)), 179)
    urls = []
    for la in range(math.floor(lat - dlat), math.floor(lat + dlat) + 1):
        for lo in range(math.floor(lon - dlon), math.floor(lon + dlon) + 1):
            lo = (lo + 180) % 360 - 180
            ns, ew = ("N" if la >= 0 else "S"), ("E" if lo >= 0 else "W")
            name = f"Copernicus_DSM_COG_30_{ns}{abs(la):02d}_00_{ew}{abs(lo):03d}_00_DEM"
            urls.append(f"{BUCKET}/{name}/{name}.tif")
    return sorted(set(urls))


def dem_on_local_grid(lat, lon, half_km, sources):
    n = int(round(2 * half_km * 1000 / CELL))
    dst = np.full((n, n), np.nan, dtype="float32")
    crs = f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
    transform = from_origin(-half_km * 1000, half_km * 1000, CELL, CELL)
    used = 0
    for s in sources:
        try:
            with rasterio.open(s) as src:
                tmp = np.full((n, n), np.nan, dtype="float32")
                reproject(rasterio.band(src, 1), tmp, dst_transform=transform, dst_crs=crs,
                          dst_nodata=np.nan, resampling=Resampling.average)
        except Exception:
            continue  # ocean tiles do not exist
        fill = np.isnan(dst) & np.isfinite(tmp)
        dst[fill] = tmp[fill]
        used += 1
    return dst, used


def hillshade(dem, az=315, alt=45):
    gy, gx = np.gradient(dem, CELL)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(-gx, gy)
    a, z = np.radians(az), np.radians(90 - alt)
    return np.cos(z) * np.cos(slope) + np.sin(z) * np.sin(slope) * np.cos(a - aspect)


def detail(site, dem_file=None, half_km=55.0, light_half_km=20.0, max_dist_m=15000):
    lat, lon = float(site["lat"]), float(site["lon"])
    sources = [dem_file] if dem_file else tile_urls(lat, lon, half_km)
    dem, used = dem_on_local_grid(lat, lon, half_km, sources)
    if used == 0 or np.isfinite(dem).mean() < 0.3:
        return None, None
    n = dem.shape[0]
    c = n // 2
    pad = int((light_half_km * 1000 + max_dist_m) / CELL)
    inner = int(light_half_km * 1000 / CELL)
    crop = dem[c - pad:c + pad, c - pad:c + pad]
    lm = lighting_metrics(crop, CELL, lat, max_dist_m=max_dist_m)
    k = pad - inner
    lit = lm["lit_frac"][k:-k, k:-k]
    lwu = lm["lit_when_up_map"][k:-k, k:-k]
    ok = np.isfinite(crop[k:-k, k:-k])
    metrics = {
        "site_code": site["site_code"], "lat": lat, "lon": lon, "dem_tiles_used": used,
        "lit_when_up": float(lwu[ok].mean()), "lit_best": float(lit[ok].max()),
        "illum_contrast": float(np.percentile(lit[ok], 95) - np.percentile(lit[ok], 5)),
        "psr_frac": float((lit[ok] == 0).mean()), "sun_up_share": lm["sun_up_share"],
        "lit_best_below_1": bool(lit[ok].max() < 1.0),
        **traffic_at_centre(dem, CELL),
    }
    b = 10000 / CELL / 2
    centre = block_stats(dem[int(c - b):int(c + b), int(c - b):int(c + b)], CELL, 10000)
    if centre:
        metrics.update({f"centre_{k}": float(v) for k, v in centre[0].items()})
    figs = {"dem": dem, "lit": lit, "inner": inner, "c": c}
    return metrics, figs


def save_figures(code, figs, out):
    dem, lit, inner, c = figs["dem"], figs["lit"], figs["inner"], figs["c"]
    sub = dem[c - inner:c + inner, c - inner:c + inner]
    ext = [-inner * CELL / 1000, inner * CELL / 1000] * 2
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    ax.imshow(hillshade(sub), cmap="gray", extent=ext)
    im = ax.imshow(lit, cmap="inferno", alpha=0.6, vmin=0, vmax=max(float(np.nanmax(lit)), 0.01), extent=ext)
    fig.colorbar(im, ax=ax, label="share of the year in direct sunlight")
    ax.set_title(f"{code}: sunlight over a year (cast shadows, 120 m)")
    ax.set_xlabel("km east"), ax.set_ylabel("km north")
    fig.tight_layout(), fig.savefig(out / "lighting.png"), plt.close(fig)

    s = slope_deg(dem, CELL)
    half = dem.shape[0] * CELL / 2000
    fig, ax = plt.subplots(figsize=(6, 5), dpi=110)
    im = ax.imshow(s, cmap="viridis", vmin=0, vmax=40, extent=[-half, half, -half, half])
    ax.contour(np.flipud(s < SLOPE_PASSABLE_DEG), levels=[0.5], colors="white", linewidths=0.4,
               extent=[-half, half, -half, half])
    for r in (2, 10, 50):
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color="#ffd166", lw=1))
    fig.colorbar(im, ax=ax, label="slope (degrees)")
    ax.set_title(f"{code}: slope, passable < {SLOPE_PASSABLE_DEG:g} deg, rings 2/10/50 km")
    ax.set_xlabel("km east"), ax.set_ylabel("km north")
    fig.tight_layout(), fig.savefig(out / "slope.png"), plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=6, help="best N sites worldwide (from the highest tiers)")
    ap.add_argument("--per-continent", type=int, default=0,
                    help="also the best N sites of each continent, e.g. 2")
    ap.add_argument("--sites", help="comma-separated site codes, e.g. MOON-0001,MOON-0004")
    ap.add_argument("--dem-file", help="use a local DEM instead of downloading tiles (testing)")
    args = ap.parse_args()
    sites = pd.read_parquet(ROOT / "data" / "processed" / "sites.parquet")
    if args.sites:
        pick = sites[sites["site_code"].isin(args.sites.split(","))]
    else:
        pick = sites[sites["tier"].isin(["Tier 1", "Tier 2"])].head(args.top)
        if len(pick) < args.top:
            pick = sites.head(args.top)
        if args.per_continent:
            per = sites.sort_values("rank").groupby("continent").head(args.per_continent)
            pick = pd.concat([pick, per]).drop_duplicates("site_code").sort_values("rank")
        print(f"dossiers for {len(pick)} sites: {', '.join(pick['site_code'])}")
    for _, site in pick.iterrows():
        out = ROOT / "docs" / "dossiers" / site["site_code"]
        out.mkdir(parents=True, exist_ok=True)
        metrics, figs = detail(site, args.dem_file)
        if metrics is None:
            print(f"{site['site_code']}: no DEM coverage, skipped")
            continue
        save_figures(site["site_code"], figs, out)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"{site['site_code']}: lit_when_up {metrics['lit_when_up']:.2f}, lit_best {metrics['lit_best']:.2f}, "
              f"psr {metrics['psr_frac']:.3f}, traffic 2/10/50 km {metrics['traffic_2km']:.2f}/"
              f"{metrics['traffic_10km']:.2f}/{metrics['traffic_50km']:.2f}")


if __name__ == "__main__":
    main()
