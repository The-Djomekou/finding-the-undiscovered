# Finding the Undiscovered

**A book of new Earth analog sites for the Moon and Mars.** NASA Space Apps Challenge 2026, challenge
"Identify Earth Locations that Analog the Permanent Moon Base Locations and Mars", Sherbrooke local event.

We searched all of Earth's land for places where astronauts and engineers could rehearse operations
for NASA's planned Moon Base at the lunar South Pole. The Artemis III Science Definition Team criteria
were translated into quantities measurable on Earth, every 10 km square of land was scored against
them, and well-known analog sites were excluded so that every result is a new candidate.

## Results (Moon base profile)

- **18,516 new candidate sites on six continents**, from 2.46 million 10 km squares of land. None lies
  within 50 km of the 32 known analog and training sites we excluded (`docs/known_analogs.csv`,
  `docs/known_analogs_extra.csv`).
- **The best analogs are at the Arctic Circle (66°N)**, where the Sun spends about 107 days a year
  low on the horizon, as at the lunar South Pole, the ground stays frozen and the terrain is easy to
  drive. The three best: **MOON-0001** (West Greenland, 66.43°N 49.54°W, score 0.86),
  **MOON-0002** (Baffin Island, Canada, 66.52°N 61.85°W, 0.85) and **MOON-0011** (East Greenland
  near Tasiilaq, 65.62°N 39.03°W, 0.84).
- **No single place has everything.** The best site differs by criterion: low sun on Baffin Island,
  Shackleton-like slopes and relief on the Andean Altiplano (Bolivia, Chile), sandy regolith in the
  Taklamakan and Sahara deserts, flat frozen ground in the New Siberian Islands. The practical answer
  is a network of complementary rehearsal sites.
- **Classic lunar analogs rank low on this profile** (Mauna Kea, Meteor Crater, Mistastin: 30th to
  40th percentile) because they were chosen for their rocks; high-latitude analogs rank higher
  (McMurdo Dry Valleys 73rd). The ranking is stable: moving any weight by 20% keeps 96 to 100 of
  the top 100 squares.

| Continent | Best site (score) | Where |
| --- | --- | --- |
| North America | MOON-0001 (0.86) | West Greenland |
| Europe | MOON-0029 (0.82) | Iceland highlands |
| Asia | MOON-0307 (0.65) | New Siberian Islands, Russia |
| South America | MOON-1184 (0.36) | Southern Patagonian Icefield edge, Chile |
| Africa | MOON-1201 (0.18) | Atlantic Sahara |
| Oceania | MOON-6485 (0.16) | Simpson Desert, Australia |
| Antarctica | none | ice-free oases are smaller than a 10 km square |

## Outputs

| Output | File |
| --- | --- |
| Executive summary (printable) | `docs/executive_summary.html` |
| The book: one page per site, one chapter per continent, dossiers for the best sites | `docs/book/index.html` |
| Interactive map: best sites per continent, Artemis priorities, one criterion at a time | `app/streamlit_app.py` (online: [app link]) |
| Every site, ranked | `data/processed/sites_moon.csv`, `data/processed/sites.parquet` |
| Best sites per continent | `docs/best_sites_by_continent.csv` |
| Results and checks | `docs/results_summary.md`, `docs/validation.md` |

To read the HTML files online, enable GitHub Pages (Settings, Pages, branch `main`, folder `/docs`);
they are then at `https://the-djomekou.github.io/finding-the-undiscovered/executive_summary.html` and
`https://the-djomekou.github.io/finding-the-undiscovered/book/`.

## How it works

| Step | Script | What it does |
| --- | --- | --- |
| 0 | `scripts/00_moon_fingerprint.py` | Shackleton crater terrain at 120 m from NASA LRO LOLA |
| 1 | `scripts/01_earth_features_gee.py` | Google Earth Engine export of 14 features per 10 km square: slope, roughness, relief, trafficability at 2, 10 and 50 km, permafrost class, sand, vegetation, aridity |
| 1b | `scripts/01b_landcover_gee.py` | Share of permanent ice and water per square (NASA MODIS land cover) |
| 2 | `scripts/02_build_table.py` | Joins the exports, removes lakes and reservoirs, adds country and continent |
| 3 | `scripts/03_score_moon.py` | Scores each criterion 0 to 1 (met at 0.70), weights them by the Artemis priorities (AHP), applies the ground rules, excludes known analogs, groups squares into sites of at most 50 km |
| 4 | `scripts/04_validate.py` | Known analogs as the method sees them; weight sensitivity |
| 5 | `scripts/05_site_detail.py` | Dossiers at 120 m: a full year of cast shadows, trafficability |
| 6 | `scripts/06_make_book.py` | The book |
| 7 | `scripts/07_executive_summary.py` | The executive summary |

Weights (Analytic Hierarchy Process, following the Artemis III priorities): illumination 0.467,
water ice 0.277, terrain and mobility 0.160, regolith and soil 0.095. A site must have no vegetation
(annual maximum NDVI below 0.2), some passable ground (at least 5% within 2 km) and at least half
ice-free land. Full details and every choice: `PIPELINE.md`.

## Run it yourself

    conda create -n undiscovered python=3.12 -y
    conda activate undiscovered
    pip install -r requirements.txt
    python -m pytest -q
    earthengine authenticate                      # once, for steps 1 and 1b
    python scripts/01_earth_features_gee.py --project YOUR-CLOUD-PROJECT
    python scripts/01b_landcover_gee.py --project YOUR-CLOUD-PROJECT
    # download the GeoTIFFs from Google Drive into data/raw/earth/ and data/raw/, then:
    python scripts/02_build_table.py
    python scripts/03_score_moon.py
    python scripts/04_validate.py
    python scripts/05_site_detail.py --top 6 --per-continent 2
    python scripts/06_make_book.py
    python scripts/07_executive_summary.py
    streamlit run app/streamlit_app.py

`bash scripts/tools/run_test_pipeline.sh` runs the whole chain on synthetic data in about 20 seconds.

## Data

NASA: LRO LOLA south-pole elevation model (Barker et al. 2023, PGDA), MODIS MOD13A2 vegetation index,
MODIS MCD12Q1 land cover, Artemis III Science Definition Team Report (2020). Partner agencies: ESA
Copernicus DEM (GLO-30 and GLO-90). Other: ERA5-Land (ECMWF, Copernicus Climate Change Service),
OpenLandMap sand content, TerraClimate, Natural Earth. Processing: Google Earth Engine, Python.

## Limits

Squares are 10 km, so small features are averaged out. Antarctica's ice-free oases are smaller than a
square. Near ice-sheet edges the frozen-ground class comes from a 9 km soil model and needs a field
check. Geology and mineralogy are not measured, and gravity, vacuum and radiation cannot be reproduced
on Earth: every site needs a visit before use. Next step: the Mars profile, scoring Earth against
NASA's 47 candidate human Exploration Zones.

## AI assistance

This project was built with help from Claude (Anthropic), a large language model. The idea, the
selection and filtering of the Artemis III criteria, and the final thresholds, weights and site size
are the team's. Claude helped write and debug the code (Earth Engine exports, scoring, validation,
shadow modelling, the book and summary generators and the Streamlit app) and draft the project text.
The team reviewed and tested everything. Each AI-assisted script starts with a line saying so. No
AI-generated images, video or audio are used.

## Team

[Names and roles]

## License

MIT
