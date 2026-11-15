# Written with assistance from Claude (Anthropic); reviewed by the team.
"""Block 3: build the book, one page per candidate site, with full dossiers for the top sites.

    python scripts/06_make_book.py
Output: docs/book/index.html (self-contained: open it in a browser, print to PDF if needed)
"""
import base64
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import artemis, scoring  # noqa: E402

DOSSIERS = ROOT / "docs" / "dossiers"
OUT = ROOT / "docs" / "book" / "index.html"
PAGES_PER_CONTINENT = 100  # full pages per continent; every other best-tier site gets a table row
TIER_COLOURS = {"Tier 1": "#e0a800", "Tier 2": "#e06c3c", "Tier 3": "#b0477a", "Tier 4": "#6c6f85"}
LIMITS = [
    "Gravity, vacuum and radiation: Earth cannot reproduce them.",
    "Cold-trap temperature: lunar cold traps are near 110 K; the lowest temperature measured on "
    "Earth is about 184 K.",
    "Micro cold traps are smaller than the 120 m grid used for shadows.",
    "Dust levitation (about 10 cm, grains near 0.1 micrometre) needs vacuum; aridity is only a proxy.",
    "Geology is not scored: the chain measures the surface, not the rock (why Mistastin scores low).",
    "The Artemis III SDT report sets no engineering thresholds: sites are ranked, not declared eligible.",
]
TESTS = {
    "s3": "Solar power, navigation and imaging under sun angles below 10 degrees for much of the year.",
    "s4": "Drilling into permanently frozen ground and testing ice extraction (ISRU) at depth.",
    "s7": "Mechanised coring to 2.5 m and drive tubes of about 70 cm in loose material.",
    "s8": "Traverses at Artemis scales: 2 km on foot, about 10 km by LTV, tens of km by pressurised rover.",
    "s2": "Landing and mobility tests on terrain whose slopes and roughness resemble Shackleton's.",
    "s1": "Handling, excavation and dust tests in sandy, loose surface material.",
}


def band(v):
    return "not scored" if pd.isna(v) else (
        "strong match" if v >= .85 else "match" if v >= .7 else "near miss" if v >= .5 else "mismatch")


def bars(row):
    out = []
    for c in artemis.CRITERION_COLS:
        v = row.get(c)
        w = 0 if pd.isna(v) else int(round(v * 100))
        colour = "#3a9d5d" if (pd.notna(v) and v >= .7) else "#c9a227" if (pd.notna(v) and v >= .5) else "#b55"
        val = "n/a" if pd.isna(v) else f"{v:.2f}"
        out.append(f'<div class="bar"><span class="lab">{artemis.CRITERION_NAMES[c]}</span>'
                   f'<span class="track"><span class="fill" style="width:{w}%;background:{colour}"></span></span>'
                   f'<span class="val">{val} · {band(v)}</span></div>')
    return "\n".join(out)


def img(path):
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" alt="{html.escape(path.stem)}">'


def fmt(v, spec=".0f", unit=""):
    return "n/a" if pd.isna(v) else f"{v:{spec}}{unit}"


def site_page(row):
    code = row["site_code"]
    met = [artemis.CRITERION_NAMES[c] for c in artemis.CRITERION_COLS if pd.notna(row[c]) and row[c] >= .7]
    miss = [artemis.CRITERION_NAMES[c] for c in artemis.CRITERION_COLS if pd.notna(row[c]) and row[c] < .5]
    near = [artemis.CRITERION_NAMES[c] for c in artemis.CRITERION_COLS if pd.notna(row[c]) and .5 <= row[c] < .7]
    tests = [TESTS[c] for c in TESTS if pd.notna(row.get(c)) and row[c] >= .7]
    d = DOSSIERS / code
    dossier = ""
    if (d / "metrics.json").exists():
        m = json.loads((d / "metrics.json").read_text())
        check = "passes" if m.get("lit_best_below_1") else "FAILS (model error)"
        dossier = f"""
<h4>Dossier measurements (Copernicus DEM, 120 m, full year of sun positions)</h4>
<table class="kv">
<tr><td>Lit when the sun is up (lit_when_up)</td><td>{m['lit_when_up']:.2f}</td></tr>
<tr><td>Best-lit point, share of the year (lit_best)</td><td>{m['lit_best']:.2f} · control &lt; 1 {check}</td></tr>
<tr><td>Rim-to-floor contrast (illum_contrast)</td><td>{m['illum_contrast']:.2f}</td></tr>
<tr><td>Never lit in the year (psr_frac)</td><td>{m['psr_frac']:.3f}</td></tr>
<tr><td>Passable within 2 / 10 / 50 km</td><td>{m['traffic_2km']:.2f} / {m['traffic_10km']:.2f} / {m['traffic_50km']:.2f}</td></tr>
</table>
<div class="figs">{img(d / 'lighting.png')}{img(d / 'slope.png')}</div>
<p class="todo">To add by hand: geological units from the national map, prior studies of this site,
access and land status.</p>"""
    tier = row["tier"] if pd.notna(row["tier"]) else "untiered"
    return f"""
<section class="site{' dossier' if dossier else ''}" id="{code}">
<h3>{html.escape(row['site'])} <span class="tier" style="background:{TIER_COLOURS.get(tier, '#999')}">{tier}</span>
<span class="score">S = {row['S']:.2f}</span></h3>
<p class="where">{row['lat']:.3f}, {row['lon']:.3f} · {html.escape(str(row['continent']))} ·
{fmt(row.get('area_km2'), ',.0f', ' km²')} · best match: {html.escape(row['target'])}</p>
<div class="cols"><div>{bars(row)}</div>
<div><table class="kv">
<tr><td>Low-sun days per year (sun up, below 10°)</td><td>{fmt(row['low_sun_days'])}</td></tr>
<tr><td>Polar-night days per year</td><td>{fmt(row['dark_days'])}</td></tr>
<tr><td>Ground ice (ERA5-Land)</td><td>{html.escape(str(row.get('ice_label', 'n/a')))}</td></tr>
<tr><td>Passable 2 / 10 / 50 km</td><td>{fmt(row.get('traffic_2km'), '.2f')} / {fmt(row.get('traffic_10km'), '.2f')} / {fmt(row.get('traffic_50km'), '.2f')}</td></tr>
<tr><td>Sand at the surface</td><td>{fmt(row.get('sand_pct'), '.0f', ' %')}</td></tr>
<tr><td>Regolith thickness</td><td>{fmt(row.get('regolith_depth_cm'), '.0f', ' cm')}</td></tr>
</table></div></div>
<p><b>Matches:</b> {', '.join(met) or 'none'} · <b>Near misses:</b> {', '.join(near) or 'none'} ·
<b>Mismatches:</b> {', '.join(miss) or 'none'}</p>
{'<p><b>What to test here:</b></p><ul>' + ''.join(f'<li>{t}</li>' for t in tests) + '</ul>' if tests else ''}
{dossier}
</section>"""


def main():
    sites = pd.read_parquet(ROOT / "data" / "processed" / "sites.parquet")
    synthetic = bool(sites.get("synthetic", pd.Series([False])).any())
    w, cr = artemis.ahp_weights()
    counts = pd.crosstab(sites["continent"], sites["tier"])
    counts["All"] = counts.sum(axis=1)
    count_rows = "".join(f"<tr><td>{html.escape(c)}</td>" + "".join(f"<td>{v}</td>" for v in r) + "</tr>"
                         for c, r in counts.iterrows())
    if "continent_rank" not in sites:
        sites["continent_rank"] = sites.sort_values("rank").groupby("continent").cumcount() + 1
    if "best_on_continent" not in sites:
        sites["best_on_continent"] = True
    order = sites.groupby("continent")["S"].max().sort_values(ascending=False).index
    chapters, shown_parts = [], []
    for cont in order:
        g = sites[sites["continent"] == cont].sort_values("continent_rank")
        pages = g.head(PAGES_PER_CONTINENT)
        rest = g.iloc[PAGES_PER_CONTINENT:]
        rest = rest[rest["best_on_continent"]]
        shown_parts.append(pages)
        best_tier = g[g["best_on_continent"]]["tier"].iloc[0] if g["best_on_continent"].any() else "n/a"
        rows = "".join(
            f"<tr><td>{int(r['continent_rank'])}</td><td>{html.escape(r['site'])}</td><td>{r['tier']}</td>"
            f"<td>{r['S']:.2f}</td><td>{r['lat']:.2f}, {r['lon']:.2f}</td><td>{int(r['n_met'])}</td></tr>"
            for _, r in rest.iterrows())
        table = (f"<p>All other {best_tier} sites on this continent ({len(rest):,}):</p>"
                 f"<table class='grid'><tr><th>#</th><th>Site</th><th>Tier</th><th>S</th><th>Lat, lon</th>"
                 f"<th>Criteria met</th></tr>{rows}</table>") if len(rest) else ""
        chapters.append(
            f"<h2 id='{html.escape(cont)}'>{html.escape(cont)}: {len(g):,} sites, best tier {best_tier}</h2>"
            + "".join(site_page(r) for _, r in pages.iterrows()) + table)
    shown = pd.concat(shown_parts) if shown_parts else sites.head(0)
    toc = " · ".join(f"<a href='#{html.escape(c)}'>{html.escape(c)}</a>" for c in order)
    known = scoring.load_known(ROOT / "docs")
    known_rows = "".join(
        f"<tr><td>{html.escape(str(k['name']))}</td><td>{html.escape(str(k['analog_for']))}</td>"
        f"<td>{k['lat']:.2f}, {k['lon']:.2f}</td>"
        f"<td>{'main list' if k['list'] == 'main' else 'extra list (approximate position)'}</td></tr>"
        for _, k in known.iterrows())
    banner = ('<div class="synthetic">SYNTHETIC TEST DATA: this book tests the pipeline. '
              'Its sites and numbers are not results.</div>') if synthetic else ""
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Finding the Undiscovered: the book</title>
<style>
body{{font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;max-width:980px;margin:0 auto;padding:24px;color:#1d1d1f;line-height:1.45}}
h1{{font-size:30px;margin-bottom:4px}} h2{{margin-top:36px;border-bottom:1px solid #ddd;padding-bottom:4px}}
.synthetic{{background:#b00020;color:#fff;padding:10px 14px;border-radius:6px;font-weight:600;margin:12px 0}}
.site{{border:1px solid #e3e3e3;border-radius:8px;padding:12px 16px;margin:14px 0;break-inside:avoid}}
.site.dossier{{border-color:#c9a227;page-break-before:always}}
.tier{{color:#fff;border-radius:10px;padding:1px 8px;font-size:13px;margin-left:6px}}
.score{{font-weight:400;color:#555;font-size:15px;margin-left:8px}} .where{{color:#555;margin-top:-6px}}
.cols{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}
.bar{{display:grid;grid-template-columns:120px 1fr 150px;gap:8px;align-items:center;font-size:13px;margin:3px 0}}
.track{{background:#eee;height:9px;border-radius:5px;overflow:hidden}} .fill{{display:block;height:9px}}
table.kv{{font-size:13px;border-collapse:collapse}} table.kv td{{padding:2px 8px 2px 0;vertical-align:top}}
table.grid{{border-collapse:collapse;font-size:14px}} table.grid td,table.grid th{{border:1px solid #ddd;padding:4px 8px}}
.figs img{{width:48%;margin:1%}} .todo{{color:#8a6d00;font-size:13px}}
</style></head><body>
{banner}
<h1>Finding the Undiscovered</h1>
<p><b>A Book of New Earth Analog Sites for the Moon and Mars.</b> Volume 1: Moon base profile
(Artemis III south pole criteria transposed to Earth). NASA Space Apps Challenge 2026.</p>
<h2>How to read a page</h2>
<p>Each criterion scores 0 to 1 and counts as met at 0.70. The tier comes from the share of scored
criteria a site meets: Tier 1 at 0.75 or more, Tier 2 at 0.50, Tier 3 at 0.25, Tier 4 below that
with at least one criterion met. The overall score S weights four groups with AHP:
illumination {w['illumination']:.3f}, ice {w['ice']:.3f}, terrain and mobility {w['terrain_mobility']:.3f},
regolith and soil {w['regolith_soil']:.3f} (consistency ratio {cr:.3f}). Sites within 50 km of a known
analog are excluded, so every page is a new candidate.</p>
<h2>Sites by continent and tier</h2>
<table class="grid"><tr><th>Continent</th>{''.join(f'<th>{c}</th>' for c in counts.columns)}</tr>{count_rows}</table>
<h2>Where every Earth analog breaks down</h2><ul>{''.join(f'<li>{x}</li>' for x in LIMITS)}</ul>
<h2>Known analogs excluded ({len(known)})</h2>
<p>No site in this book lies within 50 km of any of these. The main list is docs/known_analogs.csv;
the extra list (docs/known_analogs_extra.csv) adds well-known training and field-test sites.</p>
<table class="grid"><tr><th>Known analog</th><th>Analog for</th><th>Lat, lon</th><th>List</th></tr>{known_rows}</table>
<h2>The sites ({len(sites):,}), by continent</h2>
<p>Each continent has its own chapter, best site first: full pages for its {PAGES_PER_CONTINENT} best
sites, then a table of every other site in the highest tier reached on that continent. The complete
ranked list is in sites_moon.csv. Chapters: {toc}</p>
{''.join(chapters)}
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    n_dossiers = sum((DOSSIERS / c / "metrics.json").exists() for c in shown["site_code"])
    print(f"book: {len(shown):,} site pages, {n_dossiers} with dossiers, saved {OUT}")


if __name__ == "__main__":
    main()
