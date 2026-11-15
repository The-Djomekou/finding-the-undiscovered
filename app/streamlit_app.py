# Written with assistance from Claude (Anthropic); reviewed by the team.
"""Finding the Undiscovered: demo app. Run with:  streamlit run app/streamlit_app.py"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

SCORES = Path(__file__).resolve().parents[1] / "data" / "processed" / "sites.parquet"
CRITERIA = {"s1": "Surface material", "s2": "Terrain", "s3": "Illumination", "s4": "Water, ice",
            "s5": "Dust, wind", "s7": "Subsurface", "s8": "Mobility"}
# Ordered colours: Tier 1 brightest, Tier 4 most muted (readable on a dark background)
TIER_COLOURS = {"Tier 1": "#ffd166", "Tier 2": "#f4845f", "Tier 3": "#c75d8f", "Tier 4": "#8d8fa8"}

st.set_page_config(page_title="Finding the Undiscovered", layout="wide")
st.title("Finding the Undiscovered")
st.caption("New Earth analog sites for the Moon and Mars")


@st.cache_data
def load_sites():
    if SCORES.exists():
        return pd.read_parquet(SCORES)
    rng = np.random.default_rng(1)  # placeholder data until real scores exist
    n = 60
    df = pd.DataFrame({"site": [f"Site {i}" for i in range(n)],
                       "lat": rng.uniform(-60, 75, n), "lon": rng.uniform(-170, 170, n),
                       "body": rng.choice(["Moon", "Mars"], n),
                       "target": rng.choice(["Jezero EZ", "Malapert Massif", "Arcadia EZ"], n),
                       "tier": rng.choice(list(TIER_COLOURS), n), "synthetic": True})
    for col in CRITERIA:
        df[col] = rng.uniform(0, 1, n)
    df.loc[df["body"] == "Moon", "s5"] = np.nan  # wind does not apply on the Moon
    return df


def world_map(df):
    """Sites on an Equal Earth map, coloured by tier."""
    fig = px.scatter_geo(df, lat="lat", lon="lon", color="tier", custom_data=["site"],
                         hover_name="site",
                         hover_data={c: True for c in ["target", "tier"] if c in df}
                         | {"lat": ":.2f", "lon": ":.2f"},
                         color_discrete_map=TIER_COLOURS,
                         category_orders={"tier": list(TIER_COLOURS)},
                         projection="equal earth")
    fig.update_traces(marker={"size": 9, "line": {"width": 0.5, "color": "#111111"}})
    fig.update_geos(showland=True, landcolor="#2a2a2a", showocean=True, oceancolor="#141414",
                    showcountries=True, countrycolor="#4a4a4a", coastlinecolor="#6a6a6a",
                    showframe=False, bgcolor="rgba(0,0,0,0)")
    fig.update_layout(height=520, margin={"l": 0, "r": 0, "t": 30, "b": 0},
                      legend={"title": "", "orientation": "h", "x": 0, "y": 1.06},
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


sites = load_sites()
for col in CRITERIA:
    if col not in sites:
        sites[col] = np.nan
if "synthetic" in sites and bool(sites["synthetic"].any()):
    st.warning("Synthetic test data: these sites test the app and are not results.")
with st.sidebar:
    body = st.radio("Analog for", ["Moon", "Mars"], horizontal=True)
    has_best = "best_on_continent" in sites
    modes = (["Best sites on each continent"] if has_best else []) + ["Choose tiers"]
    mode = st.radio("Show", modes)
    if mode == "Choose tiers":
        default_tiers = list(TIER_COLOURS) if len(sites) <= 5000 else ["Tier 1", "Tier 2"]  # keep the map fast
        tiers = st.multiselect("Tiers", list(TIER_COLOURS), default=default_tiers)
    else:
        st.caption("Every site in the highest tier reached on its continent.")
    continents = sorted(sites["continent"].dropna().unique())
    chosen = st.multiselect("Continents", continents, default=continents)
keep = (sites["body"] == body) & sites["continent"].isin(chosen)
keep &= sites["best_on_continent"].astype(bool) if mode != "Choose tiers" else sites["tier"].isin(tiers)
view = sites[keep]

st.subheader(f"{len(view):,} candidate sites")
if view.empty:
    st.info(f"No {body} sites yet: run the {body} profile of the pipeline first.")
    st.stop()
event = st.plotly_chart(world_map(view), on_select="rerun", selection_mode="points",
                        key="world_map", width="stretch")
st.caption("Equal Earth projection: every region is shown at its true relative size. "
           "Click a site to open it.")

points = event.selection.points if event and event.selection else []
clicked = points[0].get("customdata", [None])[0] if points else None

if len(view):
    options = list(view["site"])
    start = options.index(clicked) if clicked in options else 0
    name = st.selectbox("Open a site", options, index=start)
    row = view[view["site"] == name].iloc[0]
    labels = [CRITERIA[c] for c in CRITERIA if pd.notna(row[c])]
    values = [row[c] for c in CRITERIA if pd.notna(row[c])]
    fig = go.Figure(go.Scatterpolar(r=values + values[:1], theta=labels + labels[:1], fill="toself"))
    fig.update_layout(polar={"radialaxis": {"range": [0, 1]}}, showlegend=False,
                      title=f"{name}: best match {row['target']}, {row['tier']}")
    st.plotly_chart(fig, width="stretch")
