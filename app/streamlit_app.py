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
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
ESRI_CREDIT = "Imagery: Esri World Imagery (Esri, Maxar, Earthstar Geographics)"
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
    names, vals, colours, texts = [], [], [], []
    for n, name, col, icon, _, _ in CRITERIA_8:
        v = row.get(col) if col else np.nan
        label, colour = band(v)
        names.append(f"{icon} {n}. {name}")
        vals.append(0 if pd.isna(v) else v)
        colours.append(colour)
        texts.append("not scored" if pd.isna(v) else f"{100 * v:.0f}%")
    fig = go.Figure(go.Bar(x=vals, y=names, orientation="h", marker_color=colours, text=texts,
                           textposition="outside", cliponaxis=False, hoverinfo="skip",
                           textfont={"size": 15}))
    fig.add_vline(x=MET, line_dash="dot", line_color="#aaaaaa", annotation_text="met at 70%",
                  annotation_position="top", annotation_font_size=14)
    fig.update_layout(height=470, margin={"l": 0, "r": 20, "t": 60, "b": 10}, showlegend=False,
                      xaxis={"range": [0, 1.22], "tickvals": [0, .5, 1], "tickformat": ".0%", "title": "",
                             "tickfont": {"size": 14}},
                      yaxis={"autorange": "reversed", "tickfont": {"size": 16}}, bargap=0.28,
                      paper_bgcolor="rgba(0,0,0,0)", font={"family": FONT},
                      title={"text": "The 8 criteria at this site", "font": {"size": 24}})
    return fig


def satellite(row, zoom=9):
    lat, lon = float(row["lat"]), float(row["lon"])
    h = CELL_DEG / 2
    w = h / max(np.cos(np.radians(lat)), 0.05)
    box_lat = [lat - h, lat - h, lat + h, lat + h, lat - h]
    box_lon = [lon - w, lon + w, lon + w, lon - w, lon - w]
    layer = {"below": "traces", "sourcetype": "raster", "sourceattribution": ESRI_CREDIT, "source": [ESRI]}
    try:  # plotly 5.24 and later (MapLibre)
        fig = go.Figure(go.Scattermap(lat=box_lat, lon=box_lon, mode="lines",
                                      line={"color": "#ffd166", "width": 2}, hoverinfo="skip"))
        fig.add_trace(go.Scattermap(lat=[lat], lon=[lon], mode="markers", marker={"size": 11, "color": "#ffd166"},
                                    hovertext=[row["site"]], hoverinfo="text"))
        fig.update_layout(map={"style": "white-bg", "center": {"lat": lat, "lon": lon}, "zoom": zoom,
                               "layers": [layer]})
    except AttributeError:  # older plotly
        fig = go.Figure(go.Scattermapbox(lat=box_lat, lon=box_lon, mode="lines",
                                         line={"color": "#ffd166", "width": 2}, hoverinfo="skip"))
        fig.add_trace(go.Scattermapbox(lat=[lat], lon=[lon], mode="markers",
                                       marker={"size": 11, "color": "#ffd166"}))
        fig.update_layout(mapbox={"style": "white-bg", "center": {"lat": lat, "lon": lon}, "zoom": zoom,
                                  "layers": [layer]})
    fig.update_layout(height=470, margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False,
                      font={"family": FONT})
    return fig


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
st.markdown("<div class='app-title'>🌍 Finding the Undiscovered</div>"
            "<div class='app-sub'>New Earth analog sites for NASA's Artemis Moon Base, scored against the "
            "Artemis III landing criteria.</div>", unsafe_allow_html=True)
with st.sidebar:
    body = st.radio("Analog for", ["Moon", "Mars"], horizontal=True)
    st.markdown("**Overall match** = weighted score of the criteria, by the Artemis priorities: "
                "illumination 47%, ice 28%, terrain and mobility 16%, regolith and soil 10%.")
    st.markdown("**A criterion is met** at a score of 70% or more.")
    if bool(sites.get("synthetic", pd.Series([False])).any()):
        st.warning("Synthetic test data: these are not results.")
if body == "Mars":
    st.info("The Mars profile (Earth scored against NASA's 47 candidate Exploration Zones) is the next step.")
    st.stop()

if "site" not in st.session_state or st.session_state["site"] not in set(sites["site_code"]):
    st.session_state["site"] = sites["site_code"].iloc[0]

k1, k2, k3, k4 = st.columns(4)
k1.metric("New candidate sites", f"{len(sites):,}")
k2.metric("Continents", sites["continent"].nunique())
k3.metric("Overall match above 70%", f"{int((sites['S'] > MET).sum()):,}")
k4.metric("Known analogs excluded (50 km)", len(known))

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
codes = list(sites["site_code"])
code = st.selectbox("🔎 Site details (or click a site above)", codes, index=codes.index(st.session_state["site"]),
                    format_func=lambda c: sites.loc[sites["site_code"] == c, "site"].iloc[0])
st.session_state["site"] = code
row = sites[sites["site_code"] == code].iloc[0]
tier_colour = TIER_COLOURS.get(row["tier"], "#999999")
st.markdown(f"### {row['site']} &nbsp;<span style='background:{tier_colour};color:#111;padding:2px 10px;"
            f"border-radius:10px;font-size:0.8em'>{row['tier']}</span>", unsafe_allow_html=True)
area = f" · {row['area_km2']:,.0f} km²" if pd.notna(row["area_km2"]) else ""
st.caption(f"{row['lat']:.2f}, {row['lon']:.2f} · {row['continent']}{area} · rank {int(row['rank']):,} of "
           f"{len(sites):,} · best match: {row['target']}")

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
    st.plotly_chart(satellite(row), width="stretch", key="sat_site")
    st.caption(f"{ESRI_CREDIT}. The yellow box is the 10 km square scored here; a site can span several squares.")


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
