#!/usr/bin/env bash
# Runs the whole Moon pipeline on SYNTHETIC inputs in a temporary copy of the repo, so your real
# data folders are never touched. Takes about a minute. Usage, from the repo root:
#     bash scripts/tools/run_test_pipeline.sh
set -euo pipefail
SRC="$(cd "$(dirname "$0")/../.." && pwd)"
TMP="$(mktemp -d)"
tar -C "$SRC" --exclude ./.git --exclude ./data/raw --exclude ./data/processed \
    --exclude ./docs/dossiers --exclude ./docs/book -cf - . | tar -C "$TMP" -xf -
cd "$TMP"
python scripts/tools/make_test_inputs.py earth 0.25
python scripts/tools/make_test_inputs.py moon
python scripts/00_moon_fingerprint.py --local data/test/synthetic_moon_dem.tif
python scripts/02_build_table.py --synthetic-test
python scripts/03_score_moon.py
python scripts/04_validate.py > /dev/null
read -r LAT LON CODE < <(python -c "import pandas as p; s=p.read_parquet('data/processed/sites.parquet').iloc[0]; print(s.lat, s.lon, s.site_code)")
python scripts/tools/make_test_inputs.py site "$LAT" "$LON"
python scripts/05_site_detail.py --sites "$CODE" --dem-file data/test/synthetic_site_dem.tif
python scripts/06_make_book.py
python -m pytest -q tests
echo
echo "Test pipeline OK. Synthetic outputs (not results) are in: $TMP"
echo "Open the test book with:  open $TMP/docs/book/index.html"
