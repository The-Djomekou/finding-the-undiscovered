"""Make SYNTHETIC inputs to test the pipeline end to end without downloads. Never for results.

    python scripts/tools/make_test_inputs.py earth [DEG]  # 2 synthetic lat-lon strips (default 0.25 deg)
    python scripts/tools/make_test_inputs.py moon         # synthetic Shackleton-like lunar DEM
    python scripts/tools/make_test_inputs.py site LAT LON # synthetic DEM around one site (EPSG:4326)
"""
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "data" / "test"
NAMES = ["slope_med_deg", "slope_iqr_deg", "roughness_m", "relief_m", "traffic_2km", "traffic_10km",
         "traffic_50km", "ice_class", "soil_t1max_k", "soil_t2max_k", "soil_t3max_k", "sand_pct",
         "ndvi_max", "aridity"]
rng = np.random.default_rng(42)


def smooth(shape, sigma):
    f = ndimage.gaussian_filter(rng.normal(size=shape), sigma)
    return (f - f.mean()) / f.std()


def earth(res=0.25):
    """Two overlapping latitude-longitude strips, like the Earth Engine export (EPSG:4326)."""
    out = ROOT / "data" / "raw" / "earth"
    out.mkdir(parents=True, exist_ok=True)
    nx, ny = int(round(360 / res)), int(round(172 / res))
    lat = 86 - (np.arange(ny) + .5) * res
    lat = np.repeat(lat[:, None], nx, axis=1)
    a = np.abs(lat)
    slope = np.clip(6 + 4 * smooth((ny, nx), 3) + rng.normal(0, 1, (ny, nx)), 0.3, 35)
    ice = np.select([a > 66, a > 60, a > 54], [3, 2, 1], 0).astype(float)
    ice = np.clip(ice + np.round(0.6 * smooth((ny, nx), 2)), 0, 3)
    tmax = 300 - 0.9 * a
    bands = {
        "slope_med_deg": slope, "slope_iqr_deg": slope * 0.8, "roughness_m": slope * 0.6 + 1,
        "relief_m": slope * 45 + 50,
        "traffic_2km": np.clip(1 - slope / 22 + 0.05 * smooth((ny, nx), 2), 0, 1),
        "traffic_10km": np.clip(1 - slope / 25, 0, 1), "traffic_50km": np.clip(1 - slope / 28, 0, 1),
        "ice_class": ice, "soil_t1max_k": tmax, "soil_t2max_k": tmax - 2, "soil_t3max_k": tmax - 4,
        "sand_pct": np.clip(50 + 20 * smooth((ny, nx), 4), 2, 98),
        "ndvi_max": np.clip(0.55 * np.exp(-((a - 15) / 28) ** 2) + 0.12 * smooth((ny, nx), 3), 0, 0.9),
        "aridity": np.clip(0.6 + 0.4 * smooth((ny, nx), 4), 0.01, 3),
    }
    stack = np.stack([bands[n] for n in NAMES]).astype("float32")
    half = nx // 2 + 1  # two overlapping strips, to test merging
    for name, c0, c1 in [("strip01", 0, half), ("strip02", half - 1, nx)]:
        t = from_origin(-180 + c0 * res, 86, res, res)
        with rasterio.open(out / f"earth_features_test_{name}.tif", "w", driver="GTiff", height=ny,
                           width=c1 - c0, count=len(NAMES), dtype="float32", crs="EPSG:4326",
                           transform=t, nodata=-9999) as dst:
            dst.write(stack[:, :, c0:c1])
    print(f"synthetic Earth strips {ny}x{nx} at {res:g} deg in {out}")


def moon():
    TEST.mkdir(parents=True, exist_ok=True)
    r = 1_737_400.0
    lon, lat = 129.20, -89.66
    rho = 2 * r * np.tan(np.radians(90 + lat) / 2)
    cx, cy = rho * np.sin(np.radians(lon)), rho * np.cos(np.radians(lon))
    cell, n = 80.0, 1000
    xs = cx - n * cell / 2 + (np.arange(n) + .5) * cell
    ys = cy + n * cell / 2 - (np.arange(n) + .5) * cell
    X, Y = np.meshgrid(xs, ys)
    d = np.hypot(X - cx, Y - cy) / 10_500.0  # Shackleton radius
    z = np.where(d < 1, -4000 * (1 - d ** 2), 400 * np.exp(-((d - 1) / 0.3) ** 2)) + 30 * smooth((n, n), 4)
    crs = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
    with rasterio.open(TEST / "synthetic_moon_dem.tif", "w", driver="GTiff", height=n, width=n, count=1,
                       dtype="float32", crs=crs, transform=from_origin(xs[0] - cell / 2, ys[0] + cell / 2, cell, cell)) as dst:
        dst.write(z.astype("float32"), 1)
    print(f"synthetic lunar DEM in {TEST / 'synthetic_moon_dem.tif'}")


def site(lat, lon):
    TEST.mkdir(parents=True, exist_ok=True)
    res = 1 / 1200  # 3 arc-seconds, like GLO-90
    dlat = 0.75
    dlon = min(0.75 / max(np.cos(np.radians(lat)), 0.1), 20)
    ny, nx = int(2 * dlat / res), int(2 * dlon / res)
    lats = lat + dlat - (np.arange(ny) + .5) * res
    lons = lon - dlon + (np.arange(nx) + .5) * res
    LO, LA = np.meshgrid(lons, lats)
    ky, kx = (LA - lat) * 111_000, (LO - lon) * 111_000 * np.cos(np.radians(lat))
    d = np.hypot(kx - 6000, ky) / 4000.0  # a 4 km radius, 600 m deep crater 6 km east of the centre
    z = 300 + 200 * smooth((ny, nx), 40) + np.where(d < 1, -600 * (1 - d ** 2), 80 * np.exp(-((d - 1) / .3) ** 2))
    with rasterio.open(TEST / "synthetic_site_dem.tif", "w", driver="GTiff", height=ny, width=nx, count=1,
                       dtype="float32", crs="EPSG:4326",
                       transform=from_origin(lons[0] - res / 2, lats[0] + res / 2, res, res)) as dst:
        dst.write(z.astype("float32"), 1)
    print(f"synthetic site DEM ({ny}x{nx}) in {TEST / 'synthetic_site_dem.tif'}")


if __name__ == "__main__":
    what = sys.argv[1]
    if what == "earth":
        earth(float(sys.argv[2]) if len(sys.argv) > 2 else 0.25)
    elif what == "moon":
        moon()
    else:
        site(float(sys.argv[2]), float(sys.argv[3]))
