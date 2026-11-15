"""Block 1, Earth (extra layer): share of permanent ice and of water in each 10 km cell.

From NASA MODIS land cover (MCD12Q1 v6.1, IGBP classes, 500 m, most common class 2019 to 2023):
  ice_frac     share of the cell classed "permanent snow and ice" (class 15)
  water_frac   share of the cell classed "water bodies" (class 17)

Why: MODIS gives no vegetation value over the Antarctic ice, so the vegetation rule dropped all
of Antarctica. With this layer, cells that are mostly ice sheet, glacier or water are removed
everywhere (they are not ground), and ice-free ground without a vegetation value counts as bare.

Run on your Mac:
    python scripts/01b_landcover_gee.py --project YOUR-CLOUD-PROJECT --check   # quick values, no export
    python scripts/01b_landcover_gee.py --project YOUR-CLOUD-PROJECT           # one global export

Then download earth_landcover_10km.tif from the Drive folder "finding-the-undiscovered" into
data/raw/ (not data/raw/earth/) and rerun 02, 03 and 04.
"""
import argparse

import ee

CRS = "EPSG:4326"
SHARD_PX = 32
CHECK_POINTS = {  # name: (lon, lat), what to expect
    "Wright Valley (Dry Valleys)": ((161.6, -77.53), "mostly ice-free: ice_frac below 1"),
    "Bunger Hills": ((100.83, -66.17), "ice-free oasis: low ice_frac"),
    "Devon Island (west)": ((-91.0, 75.4), "Arctic ground: low ice_frac"),
    "Dome C": ((123.35, -75.1), "ice sheet: ice_frac near 1"),
    "Greenland Summit": ((-38.46, 72.58), "ice sheet: ice_frac near 1"),
    "Sahara": ((5.0, 25.0), "both near 0"),
    "Lake Superior": ((-87.5, 47.7), "water_frac near 1"),
    "Sherbrooke": ((-71.89, 45.40), "low ice, low water"),
}


YEARS = ("2019-01-01", "2024-01-01")  # five complete years, 2019 to 2023


def landcover():
    # The newest year can be incomplete (it had gaps over North America), so take the most
    # common class of five complete years for each 500 m pixel.
    col = ee.ImageCollection("MODIS/061/MCD12Q1").filterDate(*YEARS).select("LC_Type1")
    proj = ee.Image(col.first()).projection()
    lc = col.mode().setDefaultProjection(proj)
    frac = ee.Image.cat([lc.eq(15).rename("ice_frac"), lc.eq(17).rename("water_frac")])
    # 500 m pixels averaged into each output cell (about 400 per 10 km cell)
    return frac.reduceResolution(ee.Reducer.mean(), maxPixels=1024), col


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True)
    ap.add_argument("--scale", type=int, default=10000)
    ap.add_argument("--shard", type=int, default=SHARD_PX)
    ap.add_argument("--check", action="store_true", help="print values at a few known places, no export")
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    img, src = landcover()
    if args.check:
        years = src.aggregate_array("system:time_start").map(lambda t: ee.Date(t).format("YYYY"))
        print("MODIS land cover years (most common class per pixel):", years.getInfo())
        for name, ((lon, lat), expect) in CHECK_POINTS.items():
            v = img.reduceRegion(ee.Reducer.first(), ee.Geometry.Point([lon, lat]), args.scale,
                                 crs=CRS).getInfo()
            ice, water = v.get("ice_frac"), v.get("water_frac")
            fmt = lambda x: "none" if x is None else f"{x:.2f}"  # noqa: E731
            print(f"  {name:27s} ice_frac {fmt(ice):>5s}  water_frac {fmt(water):>5s}   ({expect})")
        print("check done: run again without --check to export")
        return

    region = ee.Geometry.Rectangle([-180, -86, 180, 86], None, False)
    task = ee.batch.Export.image.toDrive(
        image=img.toFloat(), description="earth_landcover_10km", folder="finding-the-undiscovered",
        fileNamePrefix="earth_landcover_10km", region=region, crs=CRS, scale=args.scale,
        maxPixels=1e10, shardSize=args.shard, fileFormat="GeoTIFF")
    task.start()
    print(f"started earth_landcover_10km: task {task.id}")
    print("When it completes, download earth_landcover_10km.tif into data/raw/ and rerun 02, 03, 04.")


if __name__ == "__main__":
    main()
