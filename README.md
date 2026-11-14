# Finding the Undiscovered

A book of new Earth analog sites for the Moon and Mars. NASA Space Apps Challenge 2026, Sherbrooke local event.

## What it does

Scores every 5 km grid square of Earth's land against NASA's candidate sites on the Moon (9 Artemis regions and 4 non-polar sites) and Mars (47 Exploration Zones) on 8 analog criteria, then lists every site that meets at least one.

## Repository layout

- `data/`: raw downloads (not in git) and processed tables
- `targets/`: Moon and Mars target polygons
- `docs/`: feature schema, known analogs, dossiers
- `notebooks/`: one notebook per role
- `src/`: scoring code; `tests/`: its tests
- `app/`: Streamlit app

## Run it

    conda create -n undiscovered python=3.12 -y
    conda activate undiscovered
    pip install -r requirements.txt
    python -m pytest -q
    streamlit run app/streamlit_app.py

## Data

NASA: MODIS, SMAP, MERRA-2, NASADEM, ASTER, LRO (LOLA, Diviner, Mini-RF), Mars Odyssey, MRO. Partners: ESA Copernicus, CSA RADARSAT Constellation Mission, JAXA G-Portal.

## Team

Guilhem, Eli, Meriem, Monhel, Leea, Florian and roles (for later).

## License

MIT
