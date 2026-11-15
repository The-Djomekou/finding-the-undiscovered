# Moon pipeline: how to run it

The Moon base profile scores every land cell on Earth against the Artemis III criteria
transposed to Earth (see `docs/criteres_artemis.docx`), then groups cells into sites, checks
the ranking, measures the top sites at 120 m, and builds the book.

## 0. Set up and test (5 minutes)

```bash
conda activate undiscovered
pip install -r requirements.txt          # adds matplotlib; the rest is already installed
bash scripts/tools/run_test_pipeline.sh  # full chain on SYNTHETIC data, in a temporary folder
```

It should end with `10 passed` and `Test pipeline OK`. The test book it prints the path of is
for checking the layout only: its sites are synthetic.

## 1. Data (block 1)

| Step | Command | Time | Output |
|---|---|---|---|
| Moon fingerprint | `python scripts/00_moon_fingerprint.py` | 1 to 3 min (reads a window of a 181 MB file over the internet) | `docs/moon_fingerprint.csv` |
| Earth Engine test | `python scripts/01_earth_features_gee.py --project YOUR-PROJECT --test` | about 10 min on Google's side | one GeoTIFF in Google Drive |
| Earth Engine, whole planet | `python scripts/01_earth_features_gee.py --project YOUR-PROJECT` | 30 min to a few hours | 6 GeoTIFFs in Google Drive |
| Both in one go | `python scripts/01_earth_features_gee.py --project YOUR-PROJECT --test --all` | test first, then the 6 strips | 7 GeoTIFFs; exports use 32 px tiles (`--shard`) so they fit in memory |
| Land cover (ice, water) | `python scripts/01b_landcover_gee.py --project YOUR-PROJECT` (`--check` first) | one global export | `earth_landcover_10km.tif` into `data/raw/` |
| Table | `python scripts/02_build_table.py` | 1 to 2 min | `data/processed/earth_features.parquet` |

Earth Engine needs a Google account registered for non-commercial use
(<https://code.earthengine.google.com/register>) and `earthengine authenticate` once. Watch the
exports at <https://code.earthengine.google.com/tasks>. Download the GeoTIFFs from the Google
Drive folder `finding-the-undiscovered` into `data/raw/earth/` (remove the test file before the
full run).

Optional, criterion 7 (subsurface): regolith thickness, Pelletier et al. (2016), ORNL DAAC
dataset 1304 (free Earthdata login). Save it as `data/raw/regolith_thickness_m.tif`. Without it,
criterion 7 is skipped and the other criteria still work.

## 2. Results (block 2)

```bash
python scripts/03_score_moon.py   # scores, 50 km exclusion, constraints, sites -> docs/results_summary.md
python scripts/04_validate.py     # known analogs and weight sensitivity      -> docs/validation.md
```

## 3. Book and dossiers (block 3)

```bash
python scripts/05_site_detail.py --top 6   # Copernicus 90 m tiles from AWS, no login -> docs/dossiers/
python scripts/06_make_book.py             # -> docs/book/index.html (open it, print to PDF if needed)
streamlit run app/streamlit_app.py         # the app now reads data/processed/sites.parquet
```

## What to commit

Commit `docs/` (fingerprint, summary, validation, dossiers, book) and
`data/processed/sites.parquet`. Do not commit `data/raw/` or
`data/processed/earth_scores_moon.parquet` (large).

## Choices to confirm

- **Passable slope: 0 to 15 degrees** (team choice; the SDT report sets no threshold).
- **AHP order.** The document gives illumination 0.467 and terrain 0.160. The matrix in
  `src/artemis.py` reproduces both, which places ice second (0.277) and regolith and soil last
  (0.095).
- **Constraints.** No vegetation means annual maximum NDVI below 0.2; some flat ground means at
  least 5 % passable within 2 km.
- **Lakes.** Cells whose centre is on a lake or reservoir (Natural Earth 1:10m) are removed,
  because the elevation model shows a lake as flat bare ground. Dry salt lakes (for example
  Lake Eyre) are kept. Cells that are partly sea or lake are kept and may look flatter and less
  vegetated than the land around them.
- **Ground.** At least half of a cell must be ice-free land, neither permanent ice nor water
  (NASA MODIS land cover MCD12Q1); other cells are removed: an ice sheet or glacier is not ground. Ice-free ground with no MODIS vegetation value
  (Antarctica) counts as bare.
- **Known analogs.** 50 km around the 12 analogs in `docs/known_analogs.csv` and the 20 training
  sites in `docs/known_analogs_extra.csv` (approximate positions) is excluded.
- **Best per continent.** A continent's best sites are all its sites in the highest tier reached on
  that continent, with no fixed number (`docs/best_sites_by_continent.csv`). The book has one chapter
  per continent.
- **Site size.** A site is a group of touching cells of the same tier, at most 50 km across
  (the pressurised-rover range, SDT 4.2).
- **Regolith source.** The document names OpenLandMap; the script reads Pelletier et al. (2016)
  regolith thickness, because OpenLandMap has no depth-to-bedrock layer in Earth Engine.

## Executive summary

`python scripts/07_executive_summary.py` writes `docs/executive_summary.html`: every site scoring above 0.70, the best sites for each of the 8 criteria, the best sites on each continent with an Equal Earth map, and the method and limits. Open it in a browser and use File > Print > Save as PDF.
