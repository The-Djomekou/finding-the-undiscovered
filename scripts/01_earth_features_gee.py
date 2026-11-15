"""Block 1, Earth: export the global feature stack from Google Earth Engine.

Variables follow docs/criteres_artemis (Artemis III SDT criteria transposed to Earth):
  slope_med_deg, slope_iqr_deg   slope of the Copernicus DEM at 120 m: median and IQR per 1 km,\n                                 averaged over the output cell
  roughness_m                    std of elevation in a 3x3 window at 120 m (mean per cell)
  relief_m                       max minus min elevation at 120 m within the cell
  traffic_2km/10km/50km          share of passable terrain (slope < 15 deg) within 2, 10, 50 km
  ice_class                      3 surficial, 2 shallow (<=50 cm), 1 deep (50-100 cm), 0 dry,
                                 from ERA5-Land soil temperature: a layer whose warmest month stays
                                 below 0 C is permanently frozen
  soil_t1max_k ... soil_t3max_k  warmest-month soil temperature per layer (K), for checking
  sand_pct                       OpenLandMap sand content at the surface (%)
  ndvi_max                       MODIS annual maximum NDVI (vegetation constraint)
  aridity                        TerraClimate precipitation / potential evapotranspiration

Run on your Mac (needs an Earth Engine account, then `earthengine authenticate` once):
    python scripts/01_earth_features_gee.py --project YOUR-CLOUD-PROJECT
    python scripts/01_earth_features_gee.py --project YOUR-CLOUD-PROJECT --test   # small test region
    python scripts/01_earth_features_gee.py --project YOUR-CLOUD-PROJECT --test --all   # test + the 6 strips

Exports GeoTIFFs to the Google Drive folder "finding-the-undiscovered". When the tasks finish
(watch them at https://code.earthengine.google.com/tasks), download the files into data/raw/earth/.
"""
import argparse

import ee

SLOPE_PASSABLE_DEG = 15  # team choice: passable terrain is 0 to 15 deg (the SDT report sets none)
# Latitude-longitude everywhere: Earth Engine always accepts it and ee.Terrain handles it
# correctly. The table builder (02) converts cell sizes and areas to kilometres.
CRS = "EPSG:4326"
# Export tile size in output pixels. Each 10 km output tile pulls in every 1 km pixel under it
# (and every 120 m pixel under those). Earth Engine's default of 256 px means a 2560 km tile,
# which runs out of memory; 32 px (320 km) keeps each tile small.
SHARD_PX = 32


def terrain_bands():
    """Terrain at 120 m, then aggregated to 1 km (stage 1). Stage 2 (to the output grid) is in build_stack.

    The DEM is resampled straight to 120 m, so Earth Engine reads its pyramid level near 120 m
    instead of every 30 m pixel: chaining reduceResolution 30 m -> 120 m -> 10 km asks for more
    than 2^31 pixels per tile and fails.
    """
    col = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM")
    dem = col.mosaic().setDefaultProjection(col.first().projection())
    p120 = ee.Projection("EPSG:4326").atScale(120)
    p1k, p5k = ee.Projection(CRS).atScale(1000), ee.Projection(CRS).atScale(5000)
    dem120 = dem.resample("bilinear").reproject(p120)
    slope = ee.Terrain.slope(dem120)
    rough = dem120.reduceNeighborhood(ee.Reducer.stdDev(), ee.Kernel.square(1)).reproject(p120)
    passable = slope.lt(SLOPE_PASSABLE_DEG)

    def to1k(img, reducer):
        return img.reduceResolution(reducer, maxPixels=256).reproject(p1k)

    pass1k = to1k(passable, ee.Reducer.mean())
    pass5k = pass1k.reduceResolution(ee.Reducer.mean(), maxPixels=64).reproject(p5k)
    return {
        "slope_pct_1k": to1k(slope, ee.Reducer.percentile([25, 50, 75])),
        "rough_1k": to1k(rough, ee.Reducer.mean()),
        "minmax_1k": to1k(dem120, ee.Reducer.minMax()),
        "traffic_2km": pass1k.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(2000, "meters")).reproject(p1k),
        "traffic_10km": pass1k.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(10000, "meters")).reproject(p1k),
        "traffic_50km": pass5k.reduceNeighborhood(ee.Reducer.mean(), ee.Kernel.circle(50000, "meters")).reproject(p5k),
    }


def ice_bands():
    col = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
           .filterDate("2015-01-01", "2025-01-01")
           .select(["soil_temperature_level_1", "soil_temperature_level_2",
                    "soil_temperature_level_3"], ["t1", "t2", "t3"]))
    months = ee.List.sequence(1, 12)
    clim = ee.ImageCollection(months.map(
        lambda m: col.filter(ee.Filter.calendarRange(m, m, "month")).mean()))
    tmax = clim.max().setDefaultProjection(col.first().projection())  # warmest month, per layer
    frozen = tmax.lt(273.15)
    ice = (ee.Image(0).where(frozen.select("t3"), 1)
           .where(frozen.select("t2"), 2).where(frozen.select("t1"), 3))
    return ice.rename("ice_class"), tmax.rename(["soil_t1max_k", "soil_t2max_k", "soil_t3max_k"])


def other_bands():
    sand = ee.Image("OpenLandMap/SOL/SOL_SAND-WFRACTION_USDA-3A1A1A_M/v02").select("b0")
    modis = ee.ImageCollection("MODIS/061/MOD13A2").filterDate("2023-01-01", "2024-01-01").select("NDVI")
    # a composite has no native projection: set MODIS's, or reduceResolution refuses it
    ndvi = modis.max().multiply(0.0001).setDefaultProjection(modis.first().projection())
    tc = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE").filterDate("2015-01-01", "2025-01-01")
    years = 10
    pr = tc.select("pr").sum().divide(years)
    pet = tc.select("pet").sum().multiply(0.1).divide(years)
    aridity = pr.divide(pet.max(1)).setDefaultProjection(tc.first().projection())
    return sand.rename("sand_pct"), ndvi.rename("ndvi_max"), aridity.rename("aridity")


def build_stack(scale):
    out = ee.Projection(CRS).atScale(scale)
    t = terrain_bands()
    from_1k = min(int((scale / 1000) ** 2 * 1.5) + 16, 65535)

    def agg(img, reducer, maxp=from_1k):
        return img.reduceResolution(reducer, maxPixels=maxp).reproject(out)

    sp = t["slope_pct_1k"]  # stage 2: mean of the 1 km statistics over the output cell
    mm = t["minmax_1k"]
    ice, tmax = ice_bands()
    sand, ndvi, aridity = other_bands()
    bands = [
        agg(sp.select(1), ee.Reducer.mean()).rename("slope_med_deg"),
        agg(sp.select(2).subtract(sp.select(0)), ee.Reducer.mean()).rename("slope_iqr_deg"),
        agg(t["rough_1k"], ee.Reducer.mean()).rename("roughness_m"),
        agg(mm.select(1), ee.Reducer.max()).subtract(agg(mm.select(0), ee.Reducer.min())).rename("relief_m"),
        agg(t["traffic_2km"], ee.Reducer.mean()).rename("traffic_2km"),
        agg(t["traffic_10km"], ee.Reducer.mean()).rename("traffic_10km"),
        agg(t["traffic_50km"], ee.Reducer.mean(), 4096).rename("traffic_50km"),
        ice.reproject(out),
        tmax.reproject(out),
        sand.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(out),
        ndvi.reduceResolution(ee.Reducer.mean(), maxPixels=4096).reproject(out),
        aridity.reproject(out),
    ]
    return ee.Image.cat(bands).toFloat()


def main():
    global CRS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="your Google Cloud project registered for Earth Engine")
    ap.add_argument("--scale", type=int, default=10000, help="cell size in metres (default 10 km)")
    ap.add_argument("--crs", default=CRS, help="output projection (default EPSG:4326, latitude-longitude)")
    ap.add_argument("--bands", type=int, default=6, help="number of longitude strips to export (default 6)")
    ap.add_argument("--shard", type=int, default=SHARD_PX,
                    help="export tile size in output pixels (default 32; lower it if a task runs out of memory)")
    ap.add_argument("--test", action="store_true", help="export one small region first (Quebec-Labrador)")
    ap.add_argument("--all", action="store_true",
                    help="with --test: start the test region first, then the strips, in one run")
    ap.add_argument("--check", action="store_true",
                    help="no export: print the band names and the values at Sherbrooke, or the error")
    args = ap.parse_args()

    ee.Initialize(project=args.project)
    CRS = args.crs
    stack = build_stack(args.scale)
    if args.check:
        point = ee.Geometry.Point([-71.89, 45.40])  # Sherbrooke
        print("bands:", stack.bandNames().getInfo())
        vals = stack.reduceRegion(ee.Reducer.first(), point, args.scale, crs=CRS).getInfo()
        for k, v in vals.items():
            print(f"  {k:15s} {v}")
        # a 3 x 3 degree block is about one export tile (32 px of 10 km)
        box = ee.Geometry.Rectangle([-74, 45, -71, 48], None, False)
        area = stack.reduceRegion(ee.Reducer.mean(), box, args.scale, crs=CRS, maxPixels=1e9).getInfo()
        print(f"3x3 degree block around Sherbrooke: {sum(v is not None for v in area.values())} of "
              f"{len(area)} layers computed")
        print("check OK: run again without --check to export")
        return
    regions = []
    if args.test:
        regions.append(("test_quebec", ee.Geometry.Rectangle([-75, 45, -60, 58], None, False)))
    if not args.test or args.all:
        step = 360 / args.bands
        regions += [(f"strip{i + 1:02d}",
                     ee.Geometry.Rectangle([-180 + i * step, -86, -180 + (i + 1) * step, 86], None, False))
                    for i in range(args.bands)]
    for name, geom in regions:
        task = ee.batch.Export.image.toDrive(
            image=stack, description=f"earth_features_{name}", folder="finding-the-undiscovered",
            fileNamePrefix=f"earth_features_{args.scale // 1000}km_{name}", region=geom,
            crs=CRS, scale=args.scale, maxPixels=1e10, shardSize=args.shard, fileFormat="GeoTIFF")
        task.start()
        print(f"started {name}: task {task.id}")
    print("Watch progress at https://code.earthengine.google.com/tasks, "
          "then download the GeoTIFFs from Google Drive into data/raw/earth/")


if __name__ == "__main__":
    main()
