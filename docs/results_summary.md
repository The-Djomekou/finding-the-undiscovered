# Results summary: Moon base profile

Cells scored: 2,458,171. Cells qualifying (at least 1 criterion met, constraints passed, outside 50 km of a known analog): 236,610. Sites (touching cells of the same tier, at most 50 km across): 18,516.

Why cells drop out (a cell can fail several):

- vegetation (annual max NDVI at or above 0.2, or no NDVI value on ice-covered or unknown ground): 2,118,008
- too steep (less than 5% passable ground within 2 km): 1,076
- not ground (less than half of the cell is ice-free land, MODIS land cover): 676,618
- no criterion met: 168,019
- within 50 km of a known analog: 3,827

| Continent | Tier 1 | Tier 2 | Tier 3 | Tier 4 | All |
|---|---|---|---|---|---|
| Africa | 0 | 0 | 4461 | 1135 | 5596 |
| Asia | 0 | 0 | 3552 | 4051 | 7603 |
| Europe | 0 | 4 | 134 | 116 | 254 |
| North America | 0 | 96 | 920 | 2087 | 3103 |
| Oceania | 0 | 0 | 336 | 560 | 896 |
| South America | 0 | 0 | 313 | 751 | 1064 |

Best sites on each continent (every site in the highest tier reached there; full list in docs/best_sites_by_continent.csv):

| Continent | Best tier | Sites in it | Best site | S |
|---|---|---|---|---|
| Africa | Tier 3 | 4,461 | MOON-1201 · Morocco | 0.18 |
| Asia | Tier 3 | 3,552 | MOON-0307 · Russia | 0.65 |
| Europe | Tier 2 | 4 | MOON-0029 · Iceland | 0.82 |
| North America | Tier 2 | 96 | MOON-0001 · Greenland | 0.86 |
| Oceania | Tier 3 | 336 | MOON-6485 · Australia | 0.16 |
| South America | Tier 3 | 313 | MOON-1184 · Chile | 0.36 |
