# Validation

## Known analogs

Percentile = share of land cells with a lower overall score. Known analogs are excluded from the findings; this only checks what the method sees there.

| analog | analog_for | distance_km | S | percentile | tier | n_met | Surface material | Terrain | Illumination | Water, ice | Subsurface | Mobility |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Atacama Desert (Yungay) | Mars | 3.3 | 0.149 | 46.4 | Tier 3 | 2 | 0.78 | 0.0 | 0.0 | 0.0 |  | 0.94 |
| Haughton crater | Mars | 3.8 | 0.36 | 57.8 | Tier 4 | 1 | 0.45 | 0.0 | 0.51 | 0.0 |  | 0.97 |
| Death Valley (Badwater Basin) | Mars | 0.5 | 0.115 | 16.5 | Tier 4 | 1 | 0.48 | 0.0 | 0.0 | 0.0 |  | 0.86 |
| McMurdo Dry Valleys | Mars | 2.3 | 0.618 | 73.4 | Tier 3 | 1 |  | 0.0 | 0.51 | 1.0 |  | 0.52 |
| Holuhraun lava field | Mars | 3.8 | 0.552 | 71.1 | Tier 3 | 2 | 0.52 | 0.0 | 0.91 | 0.0 |  | 0.98 |
| Mauna Kea | Moon | 2.2 | 0.138 | 39.9 | Tier 4 | 1 | 0.53 | 0.25 | 0.0 | 0.0 |  | 0.84 |
| Mistastin Lake crater | Moon | 4.5 | 0.126 | 29.5 | Tier 4 | 1 | 0.49 | 0.0 | 0.0 | 0.0 |  | 0.99 |
| Meteor Crater | Moon | 5.4 | 0.136 | 38.0 | Tier 4 | 1 | 0.59 | 0.0 | 0.0 | 0.0 |  | 0.99 |
| Mars Desert Research Station | Mars | 1.7 | 0.131 | 34.4 | Tier 4 | 1 | 0.57 | 0.0 | 0.0 | 0.0 |  | 0.96 |
| Rio Tinto | Mars | 1.3 | 0.115 | 16.6 | Tier 4 | 1 | 0.4 | 0.0 | 0.0 | 0.0 |  | 0.95 |
| Bockfjorden volcanic complex | Mars | 0.7 | 0.476 | 66.8 | Tier 4 | 1 | 0.46 | 0.0 | 0.49 | 0.5 |  | 0.83 |
| Pilbara (Dresser Formation) | Mars | 0.6 | 0.134 | 36.4 | Tier 4 | 1 | 0.57 | 0.0 | 0.0 | 0.0 |  | 0.99 |

## Weight sensitivity

Each AHP group weight moved by 20 % down or up (then renormalised): how many of the 100 best eligible cells stay in the top 100.

| weight changed | factor | top-100 kept |
|---|---|---|
| illumination | 0.8 | 96 |
| illumination | 1.2 | 96 |
| ice | 0.8 | 100 |
| ice | 1.2 | 100 |
| terrain_mobility | 0.8 | 96 |
| terrain_mobility | 1.2 | 97 |
| regolith_soil | 0.8 | 99 |
| regolith_soil | 1.2 | 99 |
