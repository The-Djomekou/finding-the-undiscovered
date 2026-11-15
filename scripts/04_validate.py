"""Block 2: check the ranking against known analogs and test how much it depends on the weights.

    python scripts/04_validate.py
1. Known analogs: where each one ranks among all land cells (they are excluded from the
   findings, but scoring them tells us whether the method recognises known analog terrain).
   The Artemis criteria measure the surface, not the rock, so a lithology analog such as
   Mistastin can legitimately score low: an analog is an analog for a given use.
2. Weight sensitivity: each AHP weight moved by +/- 20 %, then how many of the top 100 cells
   stay in the top 100.
Output: docs/validation.md and data/processed/validation_known_analogs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import artemis, scoring  # noqa: E402

PROC = ROOT / "data" / "processed"


def md_table(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(out)


def main():
    cells = pd.read_parquet(PROC / "earth_scores_moon.parquet")
    known = pd.read_csv(ROOT / "docs" / "known_analogs.csv")
    pct = cells["S"].rank(pct=True)
    rows = []
    for _, k in known.iterrows():
        d = scoring.haversine_km(cells["lat"].to_numpy(), cells["lon"].to_numpy(), k["lat"], k["lon"])
        i = int(np.argmin(d))
        c = cells.iloc[i]
        rows.append({"analog": k["name"], "analog_for": k["analog_for"], "distance_km": round(float(d[i]), 1),
                     "S": round(float(c["S"]), 3), "percentile": round(100 * float(pct.iloc[i]), 1),
                     "tier": c["tier"], "n_met": int(c["n_met"]),
                     **{artemis.CRITERION_NAMES[s]: round(float(c[s]), 2) if pd.notna(c[s]) else None
                        for s in artemis.CRITERION_COLS}})
    res = pd.DataFrame(rows)
    res.to_csv(PROC / "validation_known_analogs.csv", index=False)

    base_w, _ = artemis.ahp_weights()
    groups = pd.DataFrame({"illumination": cells["s3"], "ice": cells["s4"],
                           "terrain_mobility": cells[["s2", "s8"]].mean(axis=1),
                           "regolith_soil": cells[["s7", "s1"]].mean(axis=1)})
    g = groups[artemis.GROUPS].to_numpy(dtype=float)
    ok = np.isfinite(g)

    def overall(w):
        wv = np.array([w[k] for k in artemis.GROUPS])
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.nansum(g * wv, axis=1) / (ok * wv).sum(axis=1)

    eligible = (cells["passes_constraints"] & cells["outside_known_buffer"]).to_numpy()
    idx = np.flatnonzero(eligible)          # rank eligible cells only
    n_top = min(100, len(idx))

    def top_set(score):
        return set(idx[np.argsort(-np.nan_to_num(score[idx], nan=-1.0))[:n_top]])

    sens = []
    if n_top:
        top = top_set(overall(base_w))
        for grp in artemis.GROUPS:
            for f in (0.8, 1.2):
                w = dict(base_w)
                w[grp] *= f
                tot = sum(w.values())
                w = {k: v / tot for k, v in w.items()}
                sens.append({"weight changed": grp, "factor": f,
                             f"top-{n_top} kept": len(top & top_set(overall(w)))})
    sens = pd.DataFrame(sens)
    sens_text = md_table(sens) if n_top else "No eligible cells in this region, so nothing to compare."

    synthetic = bool(cells["synthetic"].any())
    md = ["# Validation", ""]
    if synthetic:
        md += ["**SYNTHETIC TEST DATA: these numbers test the pipeline and are not results.**", ""]
    md += ["## Known analogs", "",
           "Percentile = share of land cells with a lower overall score. Known analogs are excluded "
           "from the findings; this only checks what the method sees there.", "",
           md_table(res), "",
           "## Weight sensitivity", "",
           "Each AHP group weight moved by 20 % down or up (then renormalised): how many of the "
           f"{n_top} best eligible cells stay in the top {n_top}.", "",
           sens_text, ""]
    (ROOT / "docs" / "validation.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
