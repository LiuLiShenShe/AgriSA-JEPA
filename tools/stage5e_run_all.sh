#!/usr/bin/env bash
# Stage 5E - full non-interactive rebuild.
#
# Prerequisites
#   * the Stage 5D ledger exists at data/corpus_v3/images/ (it is gitignored and
#     lives on this machine)
#   * the PlantDoc archive has been fetched and extracted to
#     data/holdout_raw/plantdoc_extract (see the fetch command below)
#   * PhenoBench and GWHD 2021 are at the paths recorded in
#     src/registry/stage5e_holdout_registry.py
#
# Nothing here trains anything: no JEPA forward/backward, no adapter, no probe.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv-uv/bin/python

if [ ! -d data/holdout_raw/plantdoc_extract ]; then
  echo "[stage5e] PlantDoc is not extracted; fetching it (about 1 GB)"
  mkdir -p data/holdout_raw/plantdoc data/holdout_raw/plantdoc_extract
  curl -L -o data/holdout_raw/plantdoc/plantdoc.tar.gz \
    https://codeload.github.com/pratikkayal/PlantDoc-Dataset/tar.gz/refs/heads/master
  tar -xzf data/holdout_raw/plantdoc/plantdoc.tar.gz \
    -C data/holdout_raw/plantdoc_extract --no-same-owner
fi

echo "[stage5e] 1/6 licence audit"
$PY tools/stage5e_licence_audit.py

echo "[stage5e] 2/6 holdout file-level index + cross-check"
$PY tools/stage5e_holdout_index.py --plantdoc-root data/holdout_raw/plantdoc_extract

echo "[stage5e] 3/6 pHash candidate review"
$PY tools/stage5e_hash_collision_review.py

echo "[stage5e] 4/6 build R3v3_license_verified"
$PY tools/stage5e_build_corpus.py

echo "[stage5e] 5/6 gates"
$PY tools/stage5e_gates.py

echo "[stage5e] 6/6 final report"
$PY tools/stage5e_final_report.py

echo "[stage5e] done"
