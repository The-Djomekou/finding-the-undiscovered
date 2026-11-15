"""Block 1, Moon: terrain fingerprint of Shackleton crater from the LOLA south-pole DEM.

Reads only a window around Shackleton (89.66 S, 129.20 E, diameter 21 km) from the 80 m/px
LOLA DEM (Barker et al. 2023, PGDA product 90, cloud-optimised GeoTIFF, 181 MB in full),
resamples it to 120 m, cuts it into 10 km blocks (the Earth grid cell size) and stores the
median slope, slope IQR, roughness and relief in docs/moon_fingerprint.csv.

    python scripts/00_moon_fingerprint.py                 # reads the window over the internet
    python scripts/00_moon_fingerprint.py --local LDEM_80S_80MPP_ADJ.TIF
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.terrain import block_stats  # noqa: E402

URL = "https://pgda.gsfc.nasa.gov/data/LOLA_20mpp/LDEM_80S_80MPP_ADJ.TIF"
SHACKLETON = (129.20, -89.66)  # lon, lat
MOON_R = 1_737_400.0
OUT = Path(__file__).resolve().parents[1] / "docs" / "moon_fingerprint.csv"


def south_polar_xy(lon, lat, r=MOON_R):
    """South polar stereographic on a sphere (true scale at the pole)."""
    rho = 2 * r * np.tan(np.radians(90 + lat) / 2)
    return rho * np.sin(np.radians(lon)), rho * np.cos(np.radians(lon))


def centre_xy(src, lon, lat):
    try:
        from pyproj import CRS, Transformer
        crs = CRS.from_wkt(src.crs.to_wkt())
        t = Transformer.from_crs(crs.geodetic_crs, crs, always_xy=True)
        return t.transform(lon, lat)
    except Exception:
        return south_polar_xy(lon, lat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", help="path to a downloaded LDEM_80S_80MPP_ADJ.TIF")
    ap.add_argument("--half-km", type=float, default=20.0, help="half size of the window (km)")
    args = ap.parse_args()
    path = args.local or URL
    with rasterio.open(path) as src:
        x0, y0 = centre_xy(src, *SHACKLETON)
        h = args.half_km * 1000
        win = from_bounds(x0 - h, y0 - h, x0 + h, y0 + h, src.transform)
        n = int(round(2 * h / 120))
        dem = src.read(1, window=win, out_shape=(n, n), resampling=Resampling.average,
                       boundless=True, fill_value=np.nan).astype(float)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
    print(f"window {dem.shape} at 120 m, elevation {np.nanmin(dem):.0f} to {np.nanmax(dem):.0f} m")
    rows = pd.DataFrame(block_stats(dem, 120.0, 10000))
    if rows.empty:
        sys.exit("No valid 10 km blocks: check the file and the centre coordinates.")
    fp = pd.DataFrame({"feature": rows.columns, "value": rows.median().values,
                       "iqr": (rows.quantile(0.75) - rows.quantile(0.25)).values})
    fp["n_blocks"] = len(rows)
    fp["source"] = "LOLA LDEM_80S_80MPP_ADJ (Barker et al. 2023), Shackleton +/- %g km, 120 m" % args.half_km
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fp.to_csv(OUT, index=False)
    print(fp.to_string(index=False))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
