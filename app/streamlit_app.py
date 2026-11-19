# Written with assistance from Claude (Anthropic); reviewed by the team.
"""Finding the Undiscovered: demo app. Run with:  streamlit run app/streamlit_app.py

Tabs: Map (Equal Earth), Tiers, 8 criteria, Continents. Below the tabs, the selected site: match
donuts, an 8-criteria bar chart, key values with icons, matches and what to test there, a satellite
zoom (Esri World Imagery) and the dossier figures when script 05 made them.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data" / "processed" / "sites.parquet"
FINGERPRINT = ROOT / "docs" / "moon_fingerprint.csv"
DOSSIERS = ROOT / "docs" / "dossiers"
KNOWN = [(ROOT / "docs" / "known_analogs.csv", "main list"),
         (ROOT / "docs" / "known_analogs_extra.csv", "extra list")]
BANNER = ROOT / "app" / "assets" / "banner.jpg"
BANNER_FOCUS = "center 55%"  # which part of the photo the banner shows (horizontal, vertical)
BANNER_CREDIT = "Earthset, photographed by the Artemis II crew through an Orion window on 6 April 2026. Image: NASA"
LOGOS = [ROOT / "app" / "assets" / f"space_apps_logo.{ext}" for ext in ("png", "svg", "jpg")]  # spaceappschallenge.org/brand
BODY_IMAGES = {"Moon": (ROOT / "app" / "assets" / "moon.jpg", "🌙"),
               "Mars": (ROOT / "app" / "assets" / "mars.jpg", "🔴")}
IMAGE_CREDITS = ("Moon: Galileo spacecraft, NASA/JPL. Mars: Viking 1 Orbiter mosaic, NASA/USGS. "
                 "Banner: Earthset, Artemis II, NASA.")
REPO_URL = "https://github.com/The-Djomekou/finding-the-undiscovered"
PAGES_URL = "https://the-djomekou.github.io/finding-the-undiscovered"
KPI_COLOURS = ["#1f4e79", "#2e9e6b", "#e06c3c", "#6b4f9e"]  # same as the executive summary
WEIGHTS = [("☀️ Illumination", 0.467, "#e3a800"), ("❄️ Water, ice", 0.277, "#3d8bfd"),
           ("⛰️ Terrain, mobility", 0.160, "#2eaa72"), ("🏜️ Regolith, soil", 0.095, "#c27c2c")]
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_CREDIT = "Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
S2 = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2024_3857/default/g/{z}/{y}/{x}.jpg"
S2_CREDIT = ("Sentinel-2 cloudless - <a href='https://s2maps.eu' target='_blank'>https://s2maps.eu</a> by EOX IT "
             "Services GmbH (Contains modified Copernicus Sentinel data 2024)")
MAPLIBRE = ("<link rel='stylesheet' href='https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.css' integrity="
            "'sha384-MinO0mNliZ3vwppuPOUnGa+iq619pfMhLVUXfC4LHwSCvF9H+6P/KO4Q7qBOYV5V' crossorigin='anonymous'>"
            "<script src='https://unpkg.com/maplibre-gl@4.7.1/dist/maplibre-gl.js' integrity="
            "'sha384-SYKAG6cglRMN0RVvhNeBY0r3FYKNOJtznwA0v7B5Vp9tr31xAHsZC0DqkQ/pZDmj' crossorigin='anonymous'></script>")
# Shackleton crater at 120 m (00_moon_fingerprint.py), used when the CSV is not in the repo
SHACKLETON = {"slope_med_deg": 10.38, "roughness_m": 18.02, "relief_m": 1953.7}
MET = 0.70
CELL_DEG = 10000 / 111320  # one 10 km square in degrees of latitude
TIER_COLOURS = {"Tier 1": "#ffd166", "Tier 2": "#f4845f", "Tier 3": "#c75d8f", "Tier 4": "#8d8fa8"}
TIER_RULES = {"Tier 1": "meets at least 75% of the criteria scored at the site",
              "Tier 2": "meets 50 to 75% of them",
              "Tier 3": "meets 25 to 50% of them",
              "Tier 4": "meets fewer than 25%, but at least one"}
# number, name, column, icon, what it measures, colour
CRITERIA_8 = [
    (1, "Surface material", "s1", "🏜️", "Sandy, loose surface material (sand share, OpenLandMap).", "#c27c2c"),
    (2, "Terrain like Shackleton", "s2", "⛰️", "How close slope, roughness and relief are to Shackleton crater.", "#8e6cc8"),
    (3, "Illumination", "s3", "☀️", "Days a year with the Sun up but below 10°, as at the lunar South Pole.", "#e3a800"),
    (4, "Water, ice", "s4", "❄️", "Ground frozen year-round (ERA5-Land soil temperature): surface, shallow or deep.", "#3d8bfd"),
    (5, "Dust, wind", "s5", "💨", "Not used for the Moon: it has no atmosphere.", "#8a8f98"),
    (6, "Mineralogy", None, "🪨", "Not scored: no global open mineral map at this scale.", "#8c6d62"),
    (7, "Subsurface", "s7", "🕳️", "Loose regolith deep enough to drill (needs a regolith thickness layer).", "#a0703c"),
    (8, "Mobility", "s8", "🚙", "Passable ground (slope under 15°) within 2, 10 and 50 km.", "#2eaa72"),
]
TESTS = {
    "s3": "Solar power, navigation and imaging under sun angles below 10° for much of the year.",
    "s4": "Drilling into permanently frozen ground and testing ice extraction at depth.",
    "s7": "Mechanised coring to 2.5 m and drive tubes of about 70 cm in loose material.",
    "s8": "Traverses at Artemis scales: 2 km on foot, about 10 km by rover, tens of km by pressurised rover.",
    "s2": "Landing and mobility tests on slopes and roughness like Shackleton's.",
    "s1": "Handling, excavation and dust tests in sandy, loose surface material.",
}
CONTINENT_ORDER = ["North America", "Europe", "Asia", "South America", "Africa", "Oceania", "Antarctica"]
NUMERIC = ["S", "match_ratio", "n_met", "n_scored", "s1", "s2", "s3", "s4", "s5", "s7", "s8",
           "low_sun_days", "dark_days", "ice_class", "traffic_2km", "traffic_10km", "traffic_50km",
           "sand_pct", "regolith_depth_cm", "slope_med_deg", "slope_iqr_deg", "roughness_m",
           "relief_m", "ndvi_max", "aridity", "ice_frac", "water_frac", "n_cells", "area_km2"]

st.set_page_config(page_title="Finding the Undiscovered", page_icon="🌍", layout="wide")
FONT = "Montserrat, sans-serif"
# Montserrat everywhere (also set in .streamlit/config.toml); icons keep their own font
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700;800&display=swap');
html, body, .stApp, [data-testid="stSidebar"], [data-testid="stMarkdownContainer"], [data-testid="stMetric"],
[data-testid="stMetricValue"], [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"],
.stTabs button, h1, h2, h3, h4, h5, h6, p, li, label, input, textarea {
    font-family: 'Montserrat', sans-serif;
}
.app-title { font-size: 3.8rem; font-weight: 800; line-height: 1.08; margin: 0.3rem 0 0.5rem;
             letter-spacing: -0.5px; }
.app-sub { font-size: 1.15rem; opacity: .8; margin-bottom: 1.2rem; }
@media (max-width: 700px) { .app-title { font-size: 2.4rem; } }
.banner { border-radius: 14px; padding: 2.6rem 2.2rem 1.6rem; margin: 0.2rem 0 0.4rem; min-height: 250px;
          display: flex; flex-direction: column; justify-content: flex-end; background-size: cover; }
.banner .app-title { color: #ffffff; text-shadow: 0 2px 12px rgba(0,0,0,.7); margin: 0; }
.banner .app-sub { color: #f2f2f2; opacity: 1; text-shadow: 0 1px 8px rgba(0,0,0,.8); margin: .4rem 0 0; }
.credit { font-size: .78rem; opacity: .65; margin-bottom: 1rem; }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: .4rem 0 1.2rem; }
.kpi { color: #ffffff; border-radius: 10px; padding: 12px 14px; font-size: .9rem; line-height: 1.25; }
.kpi b { display: block; font-size: 2rem; font-weight: 800; margin-bottom: 2px; }
@media (max-width: 700px) { .kpis { grid-template-columns: repeat(2, 1fr); } }
.side-h { font-weight: 700; font-size: .78rem; letter-spacing: .08em; text-transform: uppercase; opacity: .7;
          margin: .6rem 0 .3rem; }
.wrow { display: grid; grid-template-columns: 1fr 52px 34px; gap: 6px; align-items: center; font-size: .8rem;
        margin: 3px 0; }
.wtrack { background: #2b313b; height: 8px; border-radius: 4px; overflow: hidden; }
.wtrack span { display: block; height: 8px; }
.trow { display: flex; gap: 8px; align-items: baseline; font-size: .85rem; margin: 3px 0; }
.tdot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; flex: none; }
[data-testid="stSidebar"] [role="radiogroup"] img { height: 2.1em !important; max-height: none !important;
          vertical-align: middle; margin-right: .3rem; }
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- data
@st.cache_data
def load_sites():
    if SCORES.exists():
        df = pd.read_parquet(SCORES)
    else:  # placeholder until real scores exist
        rng = np.random.default_rng(1)
        n = 60
        df = pd.DataFrame({"site_code": [f"DEMO-{i:04d}" for i in range(n)],
                           "lat": rng.uniform(-60, 75, n), "lon": rng.uniform(-170, 170, n),
                           "body": "Moon", "tier": rng.choice(list(TIER_COLOURS), n),
                           "continent": rng.choice(CONTINENT_ORDER[:6], n), "country": "Demo",
                           "synthetic": True})
        for c in ["s1", "s2", "s3", "s4", "s8", "S"]:
            df[c] = rng.uniform(0, 1, n)
    for c in NUMERIC:
        if c not in df:
            df[c] = np.nan
    for c, default in [("site", None), ("country", ""), ("continent", "Unknown"), ("target", "Artemis III south pole"),
                       ("ice_label", ""), ("body", "Moon"), ("best_on_continent", True)]:
        if c not in df:
            df[c] = df["site_code"] if c == "site" else default
    if "rank" not in df:
        df["rank"] = df["S"].rank(ascending=False, method="first").astype(int)
    df["match_pct"] = (100 * df["S"]).round(0)
    return df.sort_values("rank").reset_index(drop=True)


@st.cache_data
def load_known():
    parts = [pd.read_csv(f).assign(list=lab) for f, lab in KNOWN if f.exists()]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


@st.cache_data
def shackleton():
    out = dict(SHACKLETON)
    if FINGERPRINT.exists():
        fp = pd.read_csv(FINGERPRINT)
        if {"feature", "value"} <= set(fp.columns):
            vals = dict(zip(fp["feature"], fp["value"]))
            out.update({k: float(vals[k]) for k in SHACKLETON if k in vals and pd.notna(vals[k])})
    return out


def stamp(path):
    """File modification time (0 if missing), so cached images refresh when a file is added or replaced."""
    return path.stat().st_mtime if path.exists() else 0


@st.cache_data
def chip(path_str, mtime, px=96):
    """A photo cropped to a round icon, as a data URI for widget labels; None if the file is absent."""
    if not mtime:
        return None
    import base64
    import io
    from PIL import Image, ImageDraw
    im = Image.open(path_str).convert("RGB")
    s = min(im.size)
    im = im.crop(((im.width - s) // 2, (im.height - s) // 2, (im.width + s) // 2, (im.height + s) // 2))
    im = im.resize((px, px), Image.LANCZOS)
    mask = Image.new("L", (px, px), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, px - 1, px - 1), fill=255)
    im.putalpha(mask)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data
def banner_css(mtime):
    """The banner photo as a CSS background (downscaled so the page stays light); None if absent."""
    if not mtime:
        return None
    import base64
    import io
    from PIL import Image
    im = Image.open(BANNER).convert("RGB")
    im.thumbnail((2000, 2000))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82)
    data = base64.b64encode(buf.getvalue()).decode()
    return (f"background-image: linear-gradient(180deg, rgba(0,0,0,.05) 0%, rgba(0,0,0,.55) 100%), "
            f"url(data:image/jpeg;base64,{data}); background-position: {BANNER_FOCUS};")


def scored_criteria(df):
    return [c for c in CRITERIA_8 if c[2] and c[2] in df and df[c[2]].notna().any()]


def band(v):
    if v is None or pd.isna(v):
        return "not scored", "#5a5f6a"
    if v >= .85:
        return "strong match", "#3a9d5d"
    if v >= MET:
        return "match", "#6cc070"
    if v >= .5:
        return "near miss", "#e3a800"
    return "mismatch", "#d9534f"


# ---------------------------------------------------------------- charts
def world_map(df, height=520, fit=False):
    fig = px.scatter_geo(df, lat="lat", lon="lon", color="tier", custom_data=["site_code"],
                         hover_name="site", projection="equal earth",
                         hover_data={"match_pct": ":.0f", "country": True, "lat": ":.2f", "lon": ":.2f",
                                     "tier": False},
                         labels={"match_pct": "Overall match %"},
                         color_discrete_map=TIER_COLOURS, category_orders={"tier": list(TIER_COLOURS)})
    fig.update_traces(marker={"size": 7, "line": {"width": 0.4, "color": "#111111"}})
    fig.update_geos(showland=True, landcolor="#24303f", showocean=True, oceancolor="#0f1722",
                    showcountries=True, countrycolor="#3d4c5f", coastlinecolor="#5b6b80",
                    showframe=False, bgcolor="rgba(0,0,0,0)", fitbounds="locations" if fit else False)
    fig.update_layout(height=height, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                      legend={"title": "", "orientation": "h", "x": 0, "y": 1.06},
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": FONT})
    return fig


def donut(value, label, colour, size=170):
    v = None if value is None or pd.isna(value) else float(np.clip(value, 0, 1))
    fig = go.Figure(go.Pie(values=[v or 0, 1 - (v or 0)] if v is not None else [0, 1], hole=.72, sort=False,
                           direction="clockwise", textinfo="none", hoverinfo="skip",
                           marker={"colors": [colour, "#2b313b"]}))
    fig.update_layout(showlegend=False, height=size, margin={"l": 4, "r": 4, "t": 28, "b": 4},
                      title={"text": label, "x": .5, "xanchor": "center", "font": {"size": 14}},
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": FONT},
                      annotations=[{"text": "n/a" if v is None else f"{100 * v:.0f}%", "x": .5, "y": .5,
                                    "showarrow": False, "font": {"size": 20}}])
    return fig


def criteria_bars(row):
    """Each criterion as a gauge: grey track to 100%, coloured fill, grid every 10%, value on the right."""
    names, vals, colours, texts = [], [], [], []
    for n, name, col, icon, _, _ in CRITERIA_8:
        v = row.get(col) if col else np.nan
        label, colour = band(v)
        names.append(f"{icon} {n}. {name}")
        vals.append(np.nan if pd.isna(v) else float(np.clip(v, 0, 1)))
        colours.append(colour)
        texts.append("not scored" if pd.isna(v) else f"<b>{100 * v:.0f}%</b>")
    scored = [not np.isnan(v) for v in vals]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[1] * len(names), y=names, orientation="h", hoverinfo="skip", showlegend=False,
                         marker={"color": ["#2b313b" if ok else "#1c2028" for ok in scored],
                                 "line": {"color": "#3a4250", "width": 1}}))
    fig.add_trace(go.Bar(x=[0 if np.isnan(v) else v for v in vals], y=names, orientation="h", showlegend=False,
                         marker_color=colours, hovertemplate="%{y}: %{x:.0%}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[1.04] * len(names), y=names, mode="text",
                             text=[t if ok else "" for t, ok in zip(texts, scored)], textposition="middle right",
                             textfont={"size": 16}, hoverinfo="skip", showlegend=False, cliponaxis=False))
    fig.add_trace(go.Scatter(x=[0.03] * len(names), y=names, mode="text",
                             text=["" if ok else "not scored" for ok in scored], textposition="middle right",
                             textfont={"size": 13, "color": "#8a8f98"}, hoverinfo="skip", showlegend=False))
    for i in range(1, 10):  # grid over the bars, inside the 0 to 100% track only
        fig.add_shape(type="line", x0=i / 10, x1=i / 10, y0=0, y1=1, yref="paper", layer="above",
                      line={"color": "rgba(255,255,255,0.35)" if i % 2 == 0 else "rgba(255,255,255,0.15)",
                            "width": 1})
    fig.add_vline(x=MET, line_dash="dash", line_color="#ffffff", line_width=1.5, annotation_text="met at 70%",
                  annotation_position="top", annotation_font_size=13)
    fig.update_layout(barmode="overlay", bargap=0.32, height=480, showlegend=False,
                      margin={"l": 0, "r": 10, "t": 64, "b": 10}, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font={"family": FONT},
                      title={"text": "The 8 criteria at this site", "font": {"size": 24}},
                      xaxis={"range": [0, 1.2], "tickvals": [0, .2, .4, .6, .8, 1], "tickformat": ".0%",
                             "tickfont": {"size": 13}, "showgrid": False, "zeroline": False, "title": ""},
                      yaxis={"autorange": "reversed", "tickfont": {"size": 16}})
    return fig


def embed_html(html, height):
    """Show a self-contained HTML page (st.iframe in recent Streamlit, components.html in older versions)."""
    if hasattr(st, "iframe"):
        st.iframe(html, height=height)
    else:
        import streamlit.components.v1 as components
        components.html(html, height=height)


def satellite_html(row):
    """MapLibre satellite map of the site: two small imagery tabs, metric scale bar, the 10 km square in yellow.

    Esri is drawn as two layers: a coarse one that exists everywhere (stretched when zoomed in), and a sharp one
    on top whose missing tiles are refused (blankTile=false) instead of served as grey "Map data not yet
    available" squares, so where Esri has no sharp imagery the coarse layer shows through.
    """
    lat, lon = float(row["lat"]), float(row["lon"])
    h = CELL_DEG / 2
    w = h / max(np.cos(np.radians(lat)), 0.05)
    ring = [[lon - w, lat - h], [lon + w, lat - h], [lon + w, lat + h], [lon - w, lat + h], [lon - w, lat - h]]
    cfg = json.dumps({"lat": lat, "lon": lon, "ring": ring,
                      "view": [[lon - 3 * w, lat - 3 * h], [lon + 3 * w, lat + 3 * h]],  # about 30 km across
                      "esri": ESRI, "esriCredit": ESRI_CREDIT, "s2": S2, "s2Credit": S2_CREDIT})
    return f"""<!doctype html><html><head><meta charset="utf-8">{MAPLIBRE}<style>
html, body, #map {{ height: 100%; margin: 0; background: #0e1117; }}
#map {{ border-radius: 10px; overflow: hidden; }}
#tabs {{ position: absolute; top: 8px; left: 8px; z-index: 2; display: flex; gap: 2px; padding: 2px;
        background: rgba(14,17,23,.8); border-radius: 999px; font: 600 11px Montserrat, system-ui, sans-serif; }}
#tabs button {{ border: 0; background: transparent; color: #cfd3da; padding: 3px 10px; border-radius: 999px;
               cursor: pointer; font: inherit; }}
#tabs button.on {{ background: #ffd166; color: #111; }}
.maplibregl-ctrl-scale {{ background: rgba(14,17,23,.8); color: #fff; border-color: #fff;
                         font: 600 11px Montserrat, system-ui, sans-serif; }}
.maplibregl-ctrl-attrib {{ font-size: 10px; max-width: calc(100% - 130px); }}
#err {{ color: #cfd3da; font: 13px system-ui, sans-serif; padding: 16px; }}
#overlay {{ position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; z-index: 1; }}
</style></head><body>
<div id="tabs"><button class="on" data-l="esri" title="Esri World Imagery: sharpest where available">Esri</button>
<button data-l="s2" title="Sentinel-2 cloudless 2024: 10 m everywhere">Sentinel-2</button></div>
<div id="map"></div><script>
const c = {cfg};
if (!window.maplibregl) {{
  document.getElementById('map').innerHTML = '<div id="err">The map could not load (no internet connection?).</div>';
}} else {{
  const map = new maplibregl.Map({{
    container: 'map', attributionControl: false, maxZoom: 18, dragRotate: false, pitchWithRotate: false,
    bounds: c.view,
    style: {{ version: 8,
      sources: {{
        esri_base: {{ type: 'raster', tiles: [c.esri], tileSize: 256, maxzoom: 13, attribution: c.esriCredit }},
        esri_sharp: {{ type: 'raster', tiles: [c.esri + '?blankTile=false'], tileSize: 256, minzoom: 14,
                      maxzoom: 19, attribution: c.esriCredit }},
        s2: {{ type: 'raster', tiles: [c.s2], tileSize: 256, maxzoom: 14, attribution: c.s2Credit }}
      }},
      layers: [
        {{ id: 'esri_base', type: 'raster', source: 'esri_base' }},
        {{ id: 'esri_sharp', type: 'raster', source: 'esri_sharp', minzoom: 13 }},
        {{ id: 's2', type: 'raster', source: 's2', layout: {{ visibility: 'none' }} }}
      ] }}
  }});
  // The 10 km square and the site point are drawn as an SVG overlay rather than GeoJSON layers: inside
  // Streamlit's iframe, MapLibre's web worker (which GeoJSON needs) cannot talk back to the page.
  const NS = 'http://www.w3.org/2000/svg', svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('id', 'overlay');
  const box = document.createElementNS(NS, 'polygon'), dot = document.createElementNS(NS, 'circle');
  box.setAttribute('fill', 'none'); box.setAttribute('stroke', '#ffd166'); box.setAttribute('stroke-width', '2');
  dot.setAttribute('r', '5'); dot.setAttribute('fill', '#ffd166'); dot.setAttribute('stroke', '#0e1117');
  svg.append(box, dot); map.getContainer().appendChild(svg);
  const draw = () => {{
    box.setAttribute('points', c.ring.map(p => {{ const q = map.project(p); return q.x + ',' + q.y; }}).join(' '));
    const s = map.project([c.lon, c.lat]); dot.setAttribute('cx', s.x); dot.setAttribute('cy', s.y);
  }};
  map.on('move', draw); map.on('resize', draw); draw();
  map.addControl(new maplibregl.NavigationControl({{ showCompass: false }}), 'top-right');
  map.addControl(new maplibregl.ScaleControl({{ maxWidth: 110, unit: 'metric' }}), 'bottom-left');
  map.addControl(new maplibregl.AttributionControl({{ compact: false }}), 'bottom-right');
  const LAYERS = {{ esri: ['esri_base', 'esri_sharp'], s2: ['s2'] }};
  document.querySelectorAll('#tabs button').forEach(b => b.addEventListener('click', () => {{
    document.querySelectorAll('#tabs button').forEach(x => x.classList.toggle('on', x === b));
    for (const [k, ids] of Object.entries(LAYERS))
      ids.forEach(id => map.setLayoutProperty(id, 'visibility', k === b.dataset.l ? 'visible' : 'none'));
  }}));
}}
</script></body></html>"""


# ---------------------------------------------------------------- selection helpers
def remember_click(event):
    pts = event.selection.points if event and event.selection else []
    if pts:
        st.session_state["site"] = pts[0]["customdata"][0]


def site_table(df, key, cols=None, height=300):
    cols = cols or ["site", "tier", "match_pct", "n_met", "low_sun_days", "ice_label", "sand_pct",
                    "slope_med_deg", "relief_m", "lat", "lon"]
    cols = [c for c in cols if c in df]
    show = df[cols].rename(columns={"site": "Site", "tier": "Tier", "match_pct": "Overall match %",
                                    "n_met": "Criteria met", "low_sun_days": "Low-sun days",
                                    "ice_label": "Ground ice", "sand_pct": "Sand %",
                                    "slope_med_deg": "Median slope °", "relief_m": "Relief m",
                                    "value": "Value"}).round(2)
    ev = st.dataframe(show, key=key, height=height, width="stretch", hide_index=True,
                      on_select="rerun", selection_mode="single-row")
    rows = ev.selection.rows if ev and ev.selection else []
    if rows:
        st.session_state["site"] = df.iloc[rows[0]]["site_code"]
    st.caption("Click a row (or a point on the map) to open that site below.")


def single_options(df, fp):
    opts = {f"{icon} {n}. {name}": (col, "high", None) for n, name, col, icon, _, _ in scored_criteria(df)}
    opts.update({
        f"⛰️ Slope closest to Shackleton ({fp['slope_med_deg']:.0f}° median)": ("slope_med_deg", "near", fp["slope_med_deg"]),
        f"📏 Relief closest to Shackleton ({fp['relief_m']:,.0f} m in 10 km)": ("relief_m", "near", fp["relief_m"]),
        f"🪨 Roughness closest to Shackleton ({fp['roughness_m']:.0f} m)": ("roughness_m", "near", fp["roughness_m"]),
        "☀️ Most low-sun days per year": ("low_sun_days", "high", None),
        "🏜️ Sandiest surface": ("sand_pct", "high", None),
        "🚙 Most passable ground within 10 km": ("traffic_10km", "high", None),
    })
    return {k: v for k, v in opts.items() if v[0] in df and df[v[0]].notna().any()}


def rank_single(df, col, how, target):
    d = df[df[col].notna()].copy()
    d["value"] = d[col]
    if how == "near":
        d["gap"] = (d[col] - target).abs()
        return d.sort_values(["gap", "S"], ascending=[True, False])
    return d.sort_values(["value", "S"], ascending=[False, False])


# ---------------------------------------------------------------- page
sites = load_sites()
known = load_known()
title_html = ("<div class='app-title'>🌍 Finding the Undiscovered</div>"
              "<div class='app-sub'>New Earth analog sites for NASA's Artemis Moon Base, scored against the "
              "Artemis III landing criteria.</div>")
bg = banner_css(stamp(BANNER))
if bg:
    st.markdown(f"<div class='banner' style=\"{bg}\">{title_html}</div>"
                f"<div class='credit'>{BANNER_CREDIT}</div>", unsafe_allow_html=True)
else:
    st.markdown(f"<div class='banner' style='background: linear-gradient(120deg, #0f2238 0%, #1f4e79 55%, "
                f"#2e86ab 100%)'>{title_html}</div>", unsafe_allow_html=True)

body_labels = {}
for name, (path, emoji) in BODY_IMAGES.items():
    icon = chip(str(path), stamp(path))
    body_labels[name] = f"![{name}]({icon}) {name}" if icon else f"{emoji} {name}"

with st.sidebar:
    logo = next((p for p in LOGOS if p.exists()), None)
    if logo:
        st.image(str(logo), width="stretch")
    else:
        st.markdown("## NASA Space Apps Challenge")
    st.caption("Sherbrooke local event, 2026")
    body = st.radio("Analog for", list(BODY_IMAGES), horizontal=True, format_func=body_labels.get)
    st.markdown("<div class='side-h'>Filter every view</div>", unsafe_allow_html=True)
    min_match = st.slider("Minimum overall match (%)", 0, 100, 0, step=5,
                          help="Hide sites whose overall match is below this value, on every tab.")
    st.markdown("<div class='side-h'>How sites are scored</div>", unsafe_allow_html=True)
    st.markdown("".join(f"<div class='wrow'><span>{lab}</span><span class='wtrack'><span style='width:{w * 100:.0f}%;"
                        f"background:{c}'></span></span><span>{w * 100:.0f}%</span></div>" for lab, w, c in WEIGHTS),
                unsafe_allow_html=True)
    st.caption("Overall match = weighted score, by the Artemis priorities. A criterion is met at 70% or more.")
    st.markdown("<div class='side-h'>Tiers</div>", unsafe_allow_html=True)
    st.markdown("".join(f"<div class='trow'><span class='tdot' style='background:{c}'></span><span><b>{t}</b>: "
                        f"{TIER_RULES[t]}</span></div>" for t, c in TIER_COLOURS.items()), unsafe_allow_html=True)
    st.markdown("<div class='side-h'>More</div>", unsafe_allow_html=True)
    st.markdown(f"- [📖 The site book]({PAGES_URL}/book/)\n- [📄 Executive summary]({PAGES_URL}/executive_summary.html)"
                f"\n- [💻 Code and data on GitHub]({REPO_URL})")
    with st.expander("Data sources"):
        st.markdown("NASA LRO LOLA, MODIS vegetation and land cover · ESA Copernicus DEM · ERA5-Land · "
                    "OpenLandMap · TerraClimate · Natural Earth · Esri World Imagery · Sentinel-2 cloudless by EOX. "
                    "Processed in Google Earth Engine.")
        st.caption(IMAGE_CREDITS)
    with st.expander("Team"):
        st.markdown("Eli · Meriem · Guilhem · Monhel · Leea")
    if bool(sites.get("synthetic", pd.Series([False])).any()):
        st.warning("Synthetic test data: these are not results.")
if body == "Mars":
    st.info("The Mars profile (Earth scored against NASA's 47 candidate Exploration Zones) is the next step.")
    st.stop()

if "site" not in st.session_state or st.session_state["site"] not in set(sites["site_code"]):
    st.session_state["site"] = sites["site_code"].iloc[0]

kpis = [(f"{len(sites):,}", "new candidate sites"), (f"{sites['continent'].nunique()}", "continents"),
        (f"{int((sites['S'] > MET).sum()):,}", "overall match above 70%"),
        (f"{len(known)}", "known analogs excluded (50 km)")]
st.markdown("<div class='kpis'>" + "".join(f"<div class='kpi' style='background:{c}'><b>{v}</b>{t}</div>"
                                           for (v, t), c in zip(kpis, KPI_COLOURS)) + "</div>",
            unsafe_allow_html=True)
all_sites = sites
sites = sites[sites["S"] * 100 >= min_match] if min_match else sites
if sites.empty:
    st.warning("No site reaches that minimum overall match. Lower it in the side menu.")
    st.stop()
if min_match:
    st.caption(f"Showing {len(sites):,} of {len(all_sites):,} sites (overall match {min_match}% or more).")

tab_map, tab_tiers, tab_crit, tab_cont = st.tabs(["🗺️ Map", "🏅 Tiers", "🎯 8 criteria", "🌍 Continents"])

with tab_map:
    show = st.radio("Show", ["Best sites on each continent", "Artemis priorities (light + ice)",
                             "Overall match above 70%", "All sites"], horizontal=True)
    if show == "Best sites on each continent":
        view = sites[sites["best_on_continent"].astype(bool)]
        st.caption("Every site in the highest tier reached on its continent.")
    elif show == "Artemis priorities (light + ice)":
        view = sites[(sites["s3"] >= MET) & (sites["s4"] >= MET)].sort_values("S", ascending=False)
        st.caption("Sites meeting both top Artemis priorities: illumination and water ice.")
    elif show == "Overall match above 70%":
        view = sites[sites["S"] > MET].sort_values("S", ascending=False)
    else:
        view = sites
    st.subheader(f"{len(view):,} sites")
    remember_click(st.plotly_chart(world_map(view), on_select="rerun", selection_mode="points",
                                   key="map_main", width="stretch"))
    st.caption("Equal Earth projection: every region at its true relative size.")
    site_table(view, "tbl_map")
    if len(known):
        with st.expander(f"Known analogs excluded ({len(known)}): no site lies within 50 km of these"):
            st.dataframe(known[["name", "analog_for", "lat", "lon", "list"]], width="stretch", hide_index=True)

with tab_tiers:
    counts = sites["tier"].value_counts().reindex(list(TIER_COLOURS), fill_value=0)
    st.markdown("A site's **tier** is the share of the criteria scored at that site that it meets "
                "(70% or more). The **overall match** then ranks sites inside each tier.")
    info = pd.DataFrame({"Tier": list(TIER_COLOURS), "Rule": [TIER_RULES[t] for t in TIER_COLOURS],
                         "Sites": counts.values})
    st.dataframe(info, hide_index=True, width="stretch")
    by = sites.groupby(["continent", "tier"]).size().reset_index(name="sites")
    fig = px.bar(by, x="continent", y="sites", color="tier", color_discrete_map=TIER_COLOURS,
                 category_orders={"tier": list(TIER_COLOURS)}, title="Sites per continent and tier")
    fig.update_layout(height=340, paper_bgcolor="rgba(0,0,0,0)", legend_title="", xaxis_title="",
                      font={"family": FONT})
    st.plotly_chart(fig, width="stretch", key="tier_bar")
    pick = st.radio("Open a tier", list(TIER_COLOURS), horizontal=True, index=1)
    tv = sites[sites["tier"] == pick].sort_values("S", ascending=False)
    st.subheader(f"{pick}: {len(tv):,} sites")
    if len(tv):
        remember_click(st.plotly_chart(world_map(tv, 440), on_select="rerun", selection_mode="points",
                                       key="map_tier", width="stretch"))
        site_table(tv, "tbl_tier")

with tab_crit:
    sc = scored_criteria(sites)
    met = pd.DataFrame([{"criterion": f"{icon} {n}. {name}", "sites meeting it": int((sites[col] >= MET).sum()),
                         "colour": colour} for n, name, col, icon, _, colour in sc])
    if len(met):
        fig = go.Figure(go.Bar(x=met["sites meeting it"], y=met["criterion"], orientation="h",
                               marker_color=met["colour"], text=met["sites meeting it"].map("{:,}".format),
                               textposition="outside", cliponaxis=False))
        fig.update_layout(height=300, title="How many sites meet each criterion (70% or more)",
                          yaxis={"autorange": "reversed"}, margin={"l": 0, "r": 60, "t": 40, "b": 10},
                          paper_bgcolor="rgba(0,0,0,0)", font={"family": FONT})
        st.plotly_chart(fig, width="stretch", key="crit_bar")
    with st.expander("What each criterion measures"):
        for n, name, col, icon, what, _ in CRITERIA_8:
            scored = col in sites and sites[col].notna().any() if col else False
            st.markdown(f"{icon} **{n}. {name}**: {what}" + ("" if scored else " *(not scored in this run)*"))
    opts = single_options(sites, shackleton())
    c1, c2 = st.columns([3, 1])
    choice = c1.selectbox("Rank sites by one criterion or measure (the mission weights are ignored)", list(opts))
    n_show = c2.slider("How many sites", 10, 1000, 100, step=10)
    col, how, target = opts[choice]
    if col in TESTS:
        st.info(f"What to test at these sites: {TESTS[col]}")
    cv = rank_single(sites, col, how, target).head(n_show)
    remember_click(st.plotly_chart(world_map(cv, 440), on_select="rerun", selection_mode="points",
                                   key="map_crit", width="stretch"))
    site_table(cv, "tbl_crit", ["site", "value", "tier", "match_pct", "low_sun_days", "sand_pct",
                                "slope_med_deg", "relief_m", "lat", "lon"])

with tab_cont:
    conts = [c for c in CONTINENT_ORDER if c in set(sites["continent"])] + \
            sorted(set(sites["continent"]) - set(CONTINENT_ORDER))
    labels = ["🌐 World"] + [f"{c} ({int((sites['continent'] == c).sum()):,})" for c in conts]
    where = st.radio("Continent", labels, horizontal=True)
    cont = None if where.startswith("🌐") else conts[labels.index(where) - 1]
    if "Antarctica" not in conts:
        st.caption("Antarctica has no sites: its ice-free oases are smaller than a 10 km square.")
    cv = sites if cont is None else sites[sites["continent"] == cont]
    best_tier = cv.loc[cv["best_on_continent"].astype(bool), "tier"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sites", f"{len(cv):,}")
    m2.metric("Best tier", best_tier.iloc[0] if len(best_tier) else "n/a")
    m3.metric("Best overall match", f"{cv['match_pct'].max():.0f}%")
    m4.metric("Best site", cv.sort_values("S", ascending=False)["site_code"].iloc[0])
    mapped = cv[cv["best_on_continent"].astype(bool)] if cont is None else cv
    remember_click(st.plotly_chart(world_map(mapped, 460, fit=cont is not None), on_select="rerun",
                                   selection_mode="points", key="map_cont", width="stretch"))
    rows = []
    for n, name, col, icon, _, _ in scored_criteria(cv):
        top = cv[cv[col].notna()].sort_values([col, "S"], ascending=False).head(1)
        if len(top):
            r = top.iloc[0]
            rows.append({"Criterion": f"{icon} {n}. {name}", "Best site": r["site"],
                         "Criterion score %": round(100 * r[col]), "Overall match %": r["match_pct"]})
    if rows:
        st.markdown("**Best site for each criterion here**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.markdown("**Best sites here**")
    site_table(cv.sort_values("S", ascending=False).head(200), "tbl_cont")

# ---------------------------------------------------------------- selected site
st.divider()
codes = list(all_sites["site_code"])  # every site stays reachable here, whatever the filter
names_by_code = dict(zip(all_sites["site_code"], all_sites["site"]))
code = st.selectbox("🔎 Site details (or click a site above)", codes, index=codes.index(st.session_state["site"]),
                    format_func=lambda c: names_by_code[c])
st.session_state["site"] = code
row = all_sites[all_sites["site_code"] == code].iloc[0]
tier_colour = TIER_COLOURS.get(row["tier"], "#999999")
st.markdown(f"### {row['site']} &nbsp;<span style='background:{tier_colour};color:#111;padding:2px 10px;"
            f"border-radius:10px;font-size:0.8em'>{row['tier']}</span>", unsafe_allow_html=True)
area = f" · {row['area_km2']:,.0f} km²" if pd.notna(row["area_km2"]) else ""
st.caption(f"{row['lat']:.2f}, {row['lon']:.2f} · {row['continent']}{area} · rank {int(row['rank']):,} of "
           f"{len(all_sites):,} · best match: {row['target']}")

sc = scored_criteria(sites)
ratio = row["n_met"] / row["n_scored"] if pd.notna(row["n_met"]) and row["n_scored"] else np.nan
n_met = int(row["n_met"]) if pd.notna(row["n_met"]) else "?"
n_scored = int(row["n_scored"]) if pd.notna(row["n_scored"]) else "?"
donuts = [(row["S"], "Overall match", tier_colour, "d_overall"),
          (ratio, f"Criteria met ({n_met}/{n_scored})", "#9aa5b1", "d_ratio")]
donuts += [(row[col], f"{icon} {name}", band(row[col])[1], f"d_{col}") for n, name, col, icon, _, _ in sc]
for start in range(0, len(donuts), 4):  # four per row so each ring has room
    cols = st.columns(4)
    for c, (v, label, colour, key) in zip(cols, donuts[start:start + 4]):
        c.plotly_chart(donut(v, label, colour), width="stretch", key=key)

left, right = st.columns([1.35, 1])  # criteria chart and satellite view side by side
with left:
    st.plotly_chart(criteria_bars(row), width="stretch", key="bars_site")
    st.markdown("<div style='font-size:0.9rem'><span style='color:#3a9d5d'>■</span> strong match (85%+) &nbsp; "
                "<span style='color:#6cc070'>■</span> match (70%+) &nbsp; <span style='color:#e3a800'>■</span> "
                "near miss (50%+) &nbsp; <span style='color:#d9534f'>■</span> mismatch</div>",
                unsafe_allow_html=True)
with right:
    st.markdown("#### 🛰️ Satellite view")
    embed_html(satellite_html(row), height=470)
    st.caption("Switch imagery with the tabs, top left. Where Esri has no sharp imagery, its coarser layer shows "
               "instead of grey tiles. The yellow box is the 10 km square scored here; a site can span several "
               "squares.")


def fmt(v, spec, unit=""):
    return "n/a" if v is None or pd.isna(v) else f"{v:{spec}}{unit}"


land = (1 - row["ice_frac"] - row["water_frac"]) if pd.notna(row["ice_frac"]) and pd.notna(row["water_frac"]) else np.nan
values = [
    ("☀️ Low-sun days a year", fmt(row["low_sun_days"], ".0f"), "Sun up but below 10°"),
    ("🌑 Polar-night days", fmt(row["dark_days"], ".0f"), "Sun never rises"),
    ("❄️ Ground ice", row["ice_label"] or "n/a", "ERA5-Land soil model"),
    ("💧 Aridity", fmt(row["aridity"], ".2f"), "Rain ÷ evaporation; lower is drier"),
    ("🌿 Vegetation (NDVI max)", fmt(row["ndvi_max"], ".2f"), "Below 0.2 counts as bare"),
    ("⛰️ Median slope", fmt(row["slope_med_deg"], ".1f", "°"), "Shackleton: about 10°"),
    ("📏 Relief in 10 km", fmt(row["relief_m"], ",.0f", " m"), "Shackleton: about 1,950 m"),
    ("🏜️ Sand", fmt(row["sand_pct"], ".0f", "%"), "Surface material"),
    ("🚶 Passable within 2 km", fmt(row["traffic_2km"], ".0%"), "Slope under 15°: astronaut walking range"),
    ("🚙 Passable within 10 km", fmt(row["traffic_10km"], ".0%"),
     f"Rover range; within 50 km: {fmt(row['traffic_50km'], '.0%')}"),
    ("🧊 Ice-free land", fmt(land, ".0%"), "Not ice sheet or water"),
    ("💨 Wind", "no data yet", "No wind layer in this run"),
]
for start in range(0, len(values), 4):
    cols = st.columns(4)
    for c, (label, val, help_txt) in zip(cols, values[start:start + 4]):
        c.metric(label, val, help=help_txt)

groups = {"Matches": [], "Near misses": [], "Mismatches": []}
for n, name, col, icon, _, _ in sc:
    v = row[col]
    if pd.isna(v):
        continue
    key = "Matches" if v >= MET else "Near misses" if v >= .5 else "Mismatches"
    groups[key].append(f"{icon} {name}")
g1, g2, g3 = st.columns(3)
g1.success("**Matches:** " + (", ".join(groups["Matches"]) or "none"))
g2.warning("**Near misses:** " + (", ".join(groups["Near misses"]) or "none"))
g3.error("**Mismatches:** " + (", ".join(groups["Mismatches"]) or "none"))
tests = [TESTS[col] for _, _, col, _, _, _ in sc if col in TESTS and pd.notna(row[col]) and row[col] >= MET]
if tests:
    st.markdown("**What to test here**\n" + "\n".join(f"- {t}" for t in tests))

ddir = DOSSIERS / code
if ddir.exists():
    with st.expander("📂 Dossier: 120 m detail (a full year of cast shadows, trafficability)", expanded=True):
        mfile = ddir / "metrics.json"
        if mfile.exists():
            m = json.loads(mfile.read_text())
            pretty = {k: (f"{v:.3f}" if isinstance(v, float) else str(v)) for k, v in m.items()
                      if not isinstance(v, (dict, list))}
            st.dataframe(pd.DataFrame(pretty.items(), columns=["Measure", "Value"]), hide_index=True,
                         width="stretch")
        figs = [p for p in [ddir / "lighting.png", ddir / "slope.png"] if p.exists()]
        if figs:
            fcols = st.columns(len(figs))
            for fc, p in zip(fcols, figs):
                fc.image(str(p), width="stretch")
else:
    st.caption("No dossier for this site yet (script 05 makes them for the best sites).")
