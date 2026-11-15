# Written with assistance from Claude (Anthropic); reviewed by the team.
"""Executive summary: a colourful, printable report built from the Moon results.

    python scripts/07_executive_summary.py
    open docs/executive_summary.html          # then File > Print > Save as PDF

Sections:
  1. Overall best Earth analogs: every site with an overall score above 0.70 (full book-style pages
     for the 10 best, then the complete list)
  2. Best new sites for each of the 8 criteria (each ranked on its own; full page for the winner)
  3. Best across the globe: Equal Earth map, then the best sites on each continent (full page for
     each continent's winner)
  4. Method, checks and limits

Site pages reuse the book's layout (scripts/06_make_book.py): criterion bars, matches and
mismatches, what to test there, and the dossier figures when script 05 made one.

Inputs : data/processed/sites.parquet, data/processed/validation_known_analogs.csv (optional),
         docs/known_analogs*.csv, docs/dossiers/, data/external/ne_50m_admin_0_countries.geojson
Output : docs/executive_summary.html (self-contained, one file)
"""
import base64
import html
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import artemis, scoring  # noqa: E402

_spec = importlib.util.spec_from_file_location("book", ROOT / "scripts" / "06_make_book.py")
book = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(book)  # site_page, bars, TIER_COLOURS, LIMITS, TESTS from the book

PROC = ROOT / "data" / "processed"
OUT = ROOT / "docs" / "executive_summary.html"
NE = ROOT / "data" / "external" / "ne_50m_admin_0_countries.geojson"
OVERALL_MIN = 0.70
TOP_PAGES = 10
PER_CRITERION = 5
PER_CONTINENT = 5
TIERS = book.TIER_COLOURS
CRITERIA_8 = [  # (number, name, column or None, colour, note when not scored)
    (1, "Surface material", "s1", "#c27c2c", ""),
    (2, "Terrain like Shackleton", "s2", "#6b4f9e", ""),
    (3, "Illumination", "s3", "#e3a800", ""),
    (4, "Water, ice", "s4", "#2a7fd4", ""),
    (5, "Dust, wind", None, "#8a8f98", "Not used: the Moon has no atmosphere."),
    (6, "Mineralogy", None, "#8c6d62", "Not scored: no global open mineral map at this scale."),
    (7, "Subsurface", "s7", "#7a5230", "Not scored in this run: no regolith thickness layer was added."),
    (8, "Mobility", "s8", "#2e9e6b", ""),
]
CONTINENT_COLOURS = {"North America": "#d1495b", "Europe": "#2e86ab", "Asia": "#e09f3e",
                     "South America": "#4f9d69", "Africa": "#e07a5f", "Oceania": "#00798c",
                     "Antarctica": "#8d99ae"}
KPI_COLOURS = ["#1f4e79", "#2e9e6b", "#e06c3c", "#6b4f9e"]


def esc(x):
    return html.escape("" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x))


def num(v, spec=".2f", unit=""):
    return "" if v is None or pd.isna(v) else f"{v:{spec}}{unit}"


def tier_badge(t):
    t = t if isinstance(t, str) else "untiered"
    return f'<span class="badge" style="background:{TIERS.get(t, "#999")}">{esc(t)}</span>'


def score_bar(v, colour="#1f4e79"):
    if v is None or pd.isna(v):
        return ""
    return (f'<span class="nowrap"><span class="mini"><span style="width:{max(0, min(1, v)) * 100:.0f}%;'
            f'background:{colour}"></span></span> {v:.2f}</span>')


def table(df, cols, head_colour="#1f4e79"):
    """cols: list of (header, function(row) -> html)."""
    head = "".join(f"<th style='background:{head_colour}'>{esc(h)}</th>" for h, _ in cols)
    body = "".join("<tr>" + "".join(f"<td>{f(r)}</td>" for _, f in cols) + "</tr>"
                   for _, r in df.iterrows())
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def land_share(r):
    if pd.isna(r.get("ice_frac", np.nan)) or pd.isna(r.get("water_frac", np.nan)):
        return ""
    return f"{100 * (1 - r['ice_frac'] - r['water_frac']):.0f}%"


SITE_COLS = [
    ("Site", lambda r: f"<b>{esc(r['site_code'])}</b>"),
    ("Country", lambda r: esc(r["country"])),
    ("Lat, lon", lambda r: f"{r['lat']:.2f}, {r['lon']:.2f}"),
    ("Tier", lambda r: tier_badge(r["tier"])),
    ("Overall score", lambda r: score_bar(r["S"], TIERS.get(r["tier"], "#1f4e79"))),
    ("Criteria met", lambda r: f"{int(r['n_met'])}"),
    ("Low-sun days", lambda r: num(r.get("low_sun_days"), ".0f")),
    ("Ground ice", lambda r: esc(r.get("ice_label"))),
    ("Sand", lambda r: num(r.get("sand_pct"), ".0f", "%")),
    ("Passable 10 km", lambda r: num(r.get("traffic_10km"), ".0%")),
    ("Ice-free land", land_share),
]


def card(row, shown):
    """Book-style site page in a card with a tier-coloured edge; a short pointer if already shown."""
    code = row["site_code"]
    colour = TIERS.get(row["tier"], "#999")
    if code in shown:
        return (f"<p class='seeabove'>Full page for <b>{esc(code)}</b> above "
                f"(<a href='#{esc(code)}'>jump</a>).</p>")
    shown.add(code)
    return f"<div class='card' style='border-left-color:{colour}'>{book.site_page(row)}</div>"


def section_head(n, title, colour):
    return (f"<h2 class='section' style='border-color:{colour}'><span class='num' "
            f"style='background:{colour}'>{n}</span>{esc(title)}</h2>")


def world_map(sites, over):
    """Equal Earth map: best sites on each continent by tier, sites above 0.70 outlined."""
    try:
        import geopandas as gpd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        crs = "EPSG:8857"  # WGS 84 / Equal Earth Greenwich
        fig, ax = plt.subplots(figsize=(10, 5.2))
        fig.patch.set_facecolor("#0f2238")
        ax.set_facecolor("#0f2238")
        if NE.exists():
            gpd.read_file(NE).to_crs(crs).plot(ax=ax, color="#2b4562", edgecolor="#5b7491", linewidth=0.3)
        best = sites[sites["best_on_continent"]]
        for tier, colour in TIERS.items():
            g = best[best["tier"] == tier]
            if len(g):
                pts = gpd.GeoSeries(gpd.points_from_xy(g["lon"], g["lat"]), crs="EPSG:4326").to_crs(crs)
                ax.scatter(pts.x, pts.y, s=6, color=colour, label=f"{tier} ({len(g):,})", zorder=3)
        if len(over):
            pts = gpd.GeoSeries(gpd.points_from_xy(over["lon"], over["lat"]), crs="EPSG:4326").to_crs(crs)
            ax.scatter(pts.x, pts.y, s=18, facecolors="none", edgecolors="#ffffff", linewidths=0.6,
                       label=f"Score above {OVERALL_MIN:.2f} ({len(over):,})", zorder=4)
        ax.set_axis_off()
        leg = ax.legend(loc="lower left", fontsize=8, frameon=False)
        for t in leg.get_texts():
            t.set_color("#ffffff")
        ax.set_title("Best sites on each continent (Equal Earth projection)", fontsize=10, color="#ffffff")
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return f'<img class="map" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'
    except Exception as e:  # the report still prints without the map
        print(f"map skipped: {e}")
        return ""


def main():
    sites = pd.read_parquet(PROC / "sites.parquet")
    if "best_on_continent" not in sites:
        sites["best_on_continent"] = True
    synthetic = bool(sites.get("synthetic", pd.Series([False])).any())
    known = scoring.load_known(ROOT / "docs")
    w, cr = artemis.ahp_weights()
    over = sites[sites["S"] > OVERALL_MIN].sort_values("S", ascending=False)
    continents = sites["continent"].nunique()
    shown = set()

    # 1. overall best
    by_country = (over.groupby(["continent", "country"]).agg(sites=("site_code", "size"), best=("S", "max"))
                  .reset_index().sort_values(["sites", "best"], ascending=False))
    country_table = table(by_country, [
        ("Continent", lambda r: f"<span class='dot' style='background:{CONTINENT_COLOURS.get(r['continent'], '#999')}'>"
                                f"</span>{esc(r['continent'])}"),
        ("Country", lambda r: esc(r["country"])),
        ("Sites above 0.70", lambda r: f"{r['sites']:,}"),
        ("Best score", lambda r: score_bar(r["best"]))]) if len(over) else ""
    top_cards = "".join(card(r, shown) for _, r in over.head(TOP_PAGES).iterrows())
    sec1 = (f"<p class='lead'><b>{len(over):,} new sites score above {OVERALL_MIN:.2f}</b> out of "
            f"{len(sites):,}.</p>{country_table}"
            f"<h3>The {min(TOP_PAGES, len(over))} best, in detail</h3>{top_cards}"
            f"<h3>All {len(over):,} sites above {OVERALL_MIN:.2f}</h3>"
            f"{table(over, SITE_COLS) if len(over) else '<p>No site scores above 0.70.</p>'}")

    # 2. best per criterion
    parts = []
    for n, name, col, colour, note in CRITERIA_8:
        head = (f"<h3 class='crit' style='background:{colour}'>Criterion {n}: {esc(name)}</h3>")
        if col is None or col not in sites or sites[col].notna().sum() == 0:
            parts.append(f"<div class='critbox' style='border-color:{colour}'>{head}"
                         f"<p class='note'>{esc(note or 'Not scored in this run.')}</p></div>")
            continue
        top = sites[sites[col].notna()].sort_values([col, "S"], ascending=False).head(PER_CRITERION)
        at_max = int((sites[col] >= 0.995).sum())
        met = int((sites[col] >= scoring.MET).sum())
        test = book.TESTS.get(col, "")
        cols = [("Site", lambda r: f"<b>{esc(r['site_code'])}</b>"), ("Country", lambda r: esc(r["country"])),
                ("Lat, lon", lambda r: f"{r['lat']:.2f}, {r['lon']:.2f}"),
                (f"Criterion {n}", lambda r, c=col, k=colour: score_bar(r[c], k)),
                ("Overall", lambda r: score_bar(r["S"], TIERS.get(r["tier"], "#1f4e79"))),
                ("Low-sun days", lambda r: num(r.get("low_sun_days"), ".0f")),
                ("Sand", lambda r: num(r.get("sand_pct"), ".0f", "%")),
                ("Median slope", lambda r: num(r.get("slope_med_deg"), ".1f", "°")),
                ("Relief", lambda r: num(r.get("relief_m"), ",.0f", " m"))]
        parts.append(f"<div class='critbox' style='border-color:{colour}'>{head}"
                     f"<p class='note'><b>{met:,}</b> sites meet it (0.70 or more); <b>{at_max:,}</b> score 1.00; "
                     f"ties are broken by the overall score.{' What to test: ' + esc(test) if test else ''}</p>"
                     f"{table(top, cols, colour)}{card(top.iloc[0], shown)}</div>")
    sec2 = "".join(parts)

    # 3. best across the globe
    order = list(sites.groupby("continent")["S"].max().sort_values(ascending=False).index)
    rows, blocks = [], []
    for cont in order:
        g = sites[sites["continent"] == cont].sort_values("S", ascending=False)
        bt = g[g["best_on_continent"]]
        colour = CONTINENT_COLOURS.get(cont, "#999")
        rows.append({"continent": cont, "tier": bt["tier"].iloc[0] if len(bt) else "", "n": len(bt),
                     "all": len(g), "best": g["site_code"].iloc[0], "S": g["S"].iloc[0]})
        blocks.append(f"<div class='contbox' style='border-color:{colour}'>"
                      f"<h3 class='cont' style='background:{colour}'>{esc(cont)}: {len(g):,} sites</h3>"
                      f"{table(g.head(PER_CONTINENT), SITE_COLS, colour)}{card(g.iloc[0], shown)}</div>")
    summary = table(pd.DataFrame(rows), [
        ("Continent", lambda r: f"<span class='dot' style='background:{CONTINENT_COLOURS.get(r['continent'], '#999')}'>"
                                f"</span>{esc(r['continent'])}"),
        ("Best tier", lambda r: tier_badge(r["tier"])),
        ("Sites in best tier", lambda r: f"{r['n']:,}"),
        ("All sites", lambda r: f"{r['all']:,}"),
        ("Best site", lambda r: f"<b>{esc(r['best'])}</b>"),
        ("Score", lambda r: score_bar(r["S"]))])
    missing = sorted(set(CONTINENT_COLOURS) - set(order))
    sec3 = (f"{world_map(sites, over)}{summary}"
            + (f"<p class='note'>No sites on: {', '.join(missing)}.</p>" if missing else "")
            + "".join(blocks))

    # 4. method, checks, limits
    val = PROC / "validation_known_analogs.csv"
    val_txt = ""
    if val.exists():
        v = pd.read_csv(val).sort_values("percentile", ascending=False)
        val_txt = ("<h3>Known analogs as the method sees them</h3><p class='note'>Percentile = share of land "
                   "squares with a lower overall score.</p>"
                   + table(v, [("Known analog", lambda r: esc(r["analog"])),
                               ("For", lambda r: esc(r["analog_for"])),
                               ("Overall score", lambda r: score_bar(r["S"])),
                               ("Percentile", lambda r: num(r["percentile"], ".0f"))]))
    weights = "".join(
        f"<div class='w'><span class='wl'>{esc(k.replace('_', ' and ').replace('terrain and mobility', 'terrain and mobility'))}"
        f"</span><span class='track'><span class='fill' style='width:{v * 100:.0f}%;background:{c}'></span>"
        f"</span><span class='wv'>{v:.3f}</span></div>"
        for (k, v), c in zip(w.items(), ["#e3a800", "#2a7fd4", "#2e9e6b", "#c27c2c"]))
    sec4 = f"""<p><b>Method.</b> The Artemis III Science Definition Team criteria were transposed to Earth.
Every 10 km square of land was scored from 0 to 1 on each criterion (met at 0.70) in Google Earth Engine,
from NASA LRO LOLA, NASA MODIS vegetation and land cover, ESA's Copernicus DEM, ERA5-Land soil
temperature and OpenLandMap sand content. Vegetated land, lakes, ice sheets and glaciers were removed;
touching squares of the same tier form sites of at most 50 km. All {len(known)} known analog and training
sites, and 50 km around each, were excluded.</p>
<h3>Weights from the Artemis priorities (Analytic Hierarchy Process, consistency ratio {cr:.3f})</h3>
<div class="weights">{weights}</div>
{val_txt}
<h3>Where every Earth analog breaks down</h3><ul>{''.join(f'<li>{x}</li>' for x in book.LIMITS)}</ul>
<p><b>Also:</b> squares are 10 km, so small features are averaged out; Antarctica's ice-free oases are
smaller than a square; near ice-sheet edges the frozen-ground class comes from a 9 km soil model and
needs a field check. Every site needs a visit before use.</p>"""

    banner = ("<div class='synthetic'>SYNTHETIC TEST DATA: this report tests the pipeline; its sites "
              "and numbers are not results.</div>") if synthetic else ""
    best = sites.sort_values("S", ascending=False).iloc[0]
    kpis = [(f"{len(sites):,}", "new candidate sites"), (f"{continents}", "continents"),
            (f"{len(over):,}", f"sites scoring above {OVERALL_MIN:.2f}"),
            (f"{len(known)}", "known analogs excluded (50 km)")]
    kpi_html = "".join(f"<div class='kpi' style='background:{c}'><b>{v}</b>{esc(t)}</div>"
                       for (v, t), c in zip(kpis, KPI_COLOURS))
    page = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Finding the Undiscovered: Executive Summary</title>
<style>
@page {{ size: A4; margin: 14mm 12mm; }}
* {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
body {{ font-family: -apple-system, Segoe UI, Helvetica, Arial, sans-serif; color: #1d1d1f;
       max-width: 1000px; margin: 0 auto; padding: 18px; line-height: 1.42; font-size: 13px; }}
.hero {{ background: linear-gradient(120deg, #0f2238 0%, #1f4e79 55%, #2e86ab 100%); color: #fff;
        border-radius: 10px; padding: 22px 24px; margin-bottom: 14px; }}
.hero h1 {{ font-size: 28px; margin: 0 0 6px; }} .hero p {{ margin: 0; opacity: .92; }}
.kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 14px 0; }}
.kpi {{ color: #fff; border-radius: 8px; padding: 10px 12px; font-size: 12px; }}
.kpi b {{ display: block; font-size: 24px; }}
.bestline {{ background: #fff6e0; border-left: 5px solid #e3a800; padding: 8px 12px; border-radius: 4px; }}
h2.section {{ font-size: 19px; margin-top: 26px; border-bottom: 3px solid; padding-bottom: 4px;
             page-break-before: always; page-break-after: avoid; }}
h2.section.first {{ page-break-before: auto; }}
.num {{ display: inline-block; color: #fff; border-radius: 50%; width: 26px; height: 26px; text-align: center;
        line-height: 26px; margin-right: 8px; font-size: 15px; }}
h3 {{ font-size: 14px; margin: 16px 0 6px; page-break-after: avoid; }}
h3.crit, h3.cont {{ color: #fff; padding: 6px 10px; border-radius: 6px 6px 0 0; margin: 0 0 8px; }}
.critbox, .contbox {{ border: 2px solid; border-radius: 8px; padding: 0 10px 8px; margin: 14px 0;
                      page-break-inside: auto; }}
.lead {{ font-size: 15px; }} .nowrap {{ white-space: nowrap; }} .note {{ color: #555; margin: 2px 0 8px; }}
.dot {{ display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 6px; }}
.badge {{ color: #fff; border-radius: 9px; padding: 1px 7px; font-size: 11px; white-space: nowrap; }}
.mini {{ display: inline-block; width: 46px; height: 8px; background: #e6e6e6; border-radius: 4px;
         overflow: hidden; vertical-align: middle; }} .mini span {{ display: block; height: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 6px 0 12px; font-size: 11.5px; }}
th {{ color: #fff; text-align: left; padding: 4px 6px; }}
td {{ border-bottom: 1px solid #e3e3e3; padding: 3px 6px; }} tbody tr:nth-child(even) {{ background: #f7f9fc; }}
tr {{ page-break-inside: avoid; }} thead {{ display: table-header-group; }}
.map {{ width: 100%; border-radius: 8px; margin: 6px 0 10px; }}
.card {{ border-left: 7px solid; border-radius: 8px; background: #fbfbfd; margin: 12px 0; padding: 0 2px;
         page-break-inside: avoid; }}
.seeabove {{ color: #555; font-style: italic; }}
.site {{ border: 1px solid #e3e3e3; border-radius: 8px; padding: 12px 16px; margin: 0; }}
.site h3 {{ margin-top: 2px; }} .site.dossier {{ border-color: #e3a800; }}
.tier {{ color: #fff; border-radius: 10px; padding: 1px 8px; font-size: 12px; margin-left: 6px; }}
.score {{ font-weight: 400; color: #555; font-size: 14px; margin-left: 8px; }} .where {{ color: #555; margin-top: -4px; }}
.cols {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
.bar {{ display: grid; grid-template-columns: 112px 1fr 140px; gap: 8px; align-items: center; font-size: 12px; margin: 3px 0; }}
.track {{ background: #e9e9ee; height: 9px; border-radius: 5px; overflow: hidden; }} .fill {{ display: block; height: 9px; }}
table.kv {{ font-size: 12px; width: auto; }} table.kv td {{ border: none; padding: 1px 8px 1px 0; vertical-align: top; }}
.figs img {{ width: 48%; margin: 1%; }} .todo {{ color: #8a6d00; font-size: 12px; }}
.weights .w {{ display: grid; grid-template-columns: 170px 1fr 50px; gap: 8px; align-items: center; margin: 4px 0; }}
.synthetic {{ background: #b00020; color: #fff; padding: 8px 12px; border-radius: 6px; font-weight: 600; margin-bottom: 10px; }}
</style></head><body>
{banner}
<div class="hero"><h1>Finding the Undiscovered: Executive Summary</h1>
<p>New Earth analog sites for NASA's Artemis Moon Base, scored against the Artemis III landing criteria.
NASA Space Apps Challenge 2026, Sherbrooke local event.</p></div>
<div class="kpis">{kpi_html}</div>
<p class="bestline"><b>Best overall:</b> {esc(best['site_code'])}, {esc(best['country'])}
({best['lat']:.2f}, {best['lon']:.2f}), score {best['S']:.2f}. No single place on Earth meets every
criterion, so the best plan is a network of complementary sites: see sections 2 and 3.</p>
{section_head(1, f'Overall best Earth analogs (score above {OVERALL_MIN:.2f})', '#1f4e79').replace("class='section'", "class='section first'")}{sec1}
{section_head(2, 'Best new sites for each of the 8 criteria', '#6b4f9e')}{sec2}
{section_head(3, 'Best across the globe', '#2e86ab')}{sec3}
{section_head(4, 'Method, checks and limits', '#4f9d69')}{sec4}
</body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page)
    print(f"executive summary: {len(over):,} sites above {OVERALL_MIN:.2f}, {continents} continents, "
          f"{len(shown)} full site pages, saved {OUT}")
    print("open it in a browser and print with File > Print > Save as PDF")


if __name__ == "__main__":
    main()
