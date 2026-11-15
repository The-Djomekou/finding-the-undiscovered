"""Block 1, Earth: turn the Earth Engine GeoTIFFs into one table of grid cells.

Reads every GeoTIFF in data/raw/earth/, keeps land cells, adds longitude and latitude, the
country and continent (Natural Earth 1:50m), and, if present, regolith thickness from
data/raw/regolith_thickness_m.tif (Pelletier et al. 2016, ORNL DAAC 1304, free Earthdata login).

    python scripts/02_build_table.py
Output: data/processed/earth_features.parquet
"""
import sys
from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from pyproj import CRS, Transformer

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "earth"
OUT = ROOT / "data" / "processed" / "earth_features.parquet"
NE = ROOT / "data" / "external" / "ne_50m_admin_0_countries.geojson"
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
          "ne_50m_admin_0_countries.geojson")
LAKES = ROOT / "data" / "external" / "ne_10m_lakes.geojson"
LAKES_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/"
             "ne_10m_lakes.geojson")
REGOLITH = ROOT / "data" / "raw" / "regolith_thickness_m.tif"
LANDCOVER = ROOT / "data" / "raw" / "earth_landcover_10km.tif"  # from 01b_landcover_gee.py
BAND_NAMES = ["slope_med_deg", "slope_iqr_deg", "roughness_m", "relief_m", "traffic_2km",
              "traffic_10km", "traffic_50km", "ice_class", "soil_t1max_k", "soil_t2max_k",
              "soil_t3max_k", "sand_pct", "ndvi_max", "aridity"]


def read_tif(path):
    with rasterio.open(path) as src:
        data = src.read().astype("float32")
        names = list(src.descriptions)
        if not all(names) or len(set(names)) != len(names):
            if src.count != len(BAND_NAMES):
                sys.exit(f"{path.name}: {src.count} bands, expected {len(BAND_NAMES)}")
            names = BAND_NAMES
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        rows, cols = np.nonzero(np.isfinite(data[0]))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        df = pd.DataFrame({"x": np.asarray(xs), "y": np.asarray(ys)})
        for i, n in enumerate(names):
            df[n] = data[i, rows, cols]
        res = abs(src.transform.a)
        df.attrs["crs"] = src.crs.to_wkt()
    return df, res


def add_place(df):
    if not NE.exists():
        NE.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(NE_URL, NE)
    countries = gpd.read_file(NE)[["NAME", "CONTINENT", "geometry"]]
    pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
    joined = gpd.sjoin(pts, countries, how="inner", predicate="within")
    joined = joined[~joined.index.duplicated() & (joined["CONTINENT"] != "Seven seas (open ocean)")]
    out = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
    out = out.rename(columns={"NAME": "country", "CONTINENT": "continent"})
    # Natural Earth files whole countries under one continent: fix the two large cases
    ru = out["country"].eq("Russia") & ((out["lon"] > 60) | (out["lon"] < -160))
    out.loc[ru, "continent"] = "Asia"
    gf = out["country"].eq("France") & (out["lat"] < 15) & (out["lon"] < 0)
    out.loc[gf, "continent"] = "South America"
    return out


def drop_lakes(df):
    """Remove cells whose centre is on a lake or reservoir (Natural Earth 1:10m lakes).

    The country outlines count lakes as land, and the elevation model shows a lake as perfectly
    flat ground with no vegetation, so lakes would otherwise pass as smooth, bare analog sites.
    Salt lakes are kept when they are dry (annual max NDVI at or above 0), because a dry salt
    lake such as Lake Eyre is real ground; a wet one (NDVI below 0) is removed.
    """
    if not LAKES.exists():
        LAKES.parent.mkdir(parents=True, exist_ok=True)
        urlretrieve(LAKES_URL, LAKES)
    lakes = gpd.read_file(LAKES)[["featurecla", "geometry"]]
    pts = gpd.GeoDataFrame(df[["ndvi_max"]], geometry=gpd.points_from_xy(df["lon"], df["lat"]),
                           crs="EPSG:4326")
    j = gpd.sjoin(pts, lakes, how="left", predicate="within")
    j = j[~j.index.duplicated()].reindex(df.index)
    wet = j["featurecla"].notna() & ((j["featurecla"] != "Alkaline Lake") | (j["ndvi_max"] < 0))
    print(f"removed {int(wet.sum()):,} cells on lakes and reservoirs")
    return df[~wet.to_numpy()]


def add_landcover(df):
    """Share of permanent ice and of water per cell (MODIS land cover), sampled at the cell centre."""
    if not LANDCOVER.exists():
        print("no land-cover file: ice sheets and Antarctica are handled by the vegetation rule only")
        return df
    with rasterio.open(LANDCOVER) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = t.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
        vals = np.array(list(src.sample(zip(xs, ys))), dtype=float)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    df["ice_frac"], df["water_frac"] = vals[:, 0], vals[:, 1]
    print(f"land cover added: {int((df['ice_frac'] >= 0.5).sum()):,} cells mostly ice, "
          f"{int((df['water_frac'] >= 0.5).sum()):,} mostly water")
    return df


def add_regolith(df):
    if not REGOLITH.exists():
        print("no regolith file: criterion 7 (subsurface) will be skipped")
        return df
    with rasterio.open(REGOLITH) as src:
        t = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = t.transform(df["lon"].to_numpy(), df["lat"].to_numpy())
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype=float)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    df["regolith_depth_cm"] = vals * 100
    return df


def main():
    synthetic = "--synthetic-test" in sys.argv  # only for the pipeline test, never for results
    tifs = sorted(RAW.glob("*.tif"))
    if not tifs:
        sys.exit(f"No GeoTIFFs in {RAW}: download the Earth Engine exports there first.")
    parts, res = zip(*(read_tif(p) for p in tifs))
    crs = parts[0].attrs["crs"]  # the projection the GeoTIFFs were exported in
    df = pd.concat(parts, ignore_index=True).drop_duplicates(["x", "y"])
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        df["x"].to_numpy(), df["y"].to_numpy())
    df["lon"], df["lat"] = lon, lat
    df = add_place(df)
    df = drop_lakes(df)
    df = add_landcover(df)
    df = add_regolith(df)
    df.insert(0, "site_id", [f"E{i:07d}" for i in range(len(df))])
    df["grid_m"] = res[0]  # cell size in the file's own units (degrees for EPSG:4326)
    if CRS.from_wkt(crs).is_geographic:
        df["cell_area_km2"] = (res[0] * 111.32) ** 2 * np.cos(np.radians(df["lat"]))
    else:
        df["cell_area_km2"] = (res[0] / 1000) ** 2
    df["synthetic"] = synthetic
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"{len(df):,} land cells (median cell {df['cell_area_km2'].median():.0f} km2), saved {OUT}")
    print(df.groupby("continent").size().to_string())


if __name__ == "__main__":
    main()
