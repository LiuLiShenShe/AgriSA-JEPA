#!/usr/bin/env bash
# Stage 5E.1 - evidence reconciliation rebuild (non-interactive).
#
# This stage repairs evidence and reports only.  It does NOT retrain, does not
# run an optimizer step, does not re-derive R3v3 membership, and does not touch
# any Stage 5C/5D file.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=.venv-uv/bin/python

echo "[stage5e1] 1/9 holdout file-level index rebuild (decode-status fix)"
$PY tools/stage5e_holdout_index.py --plantdoc-root data/holdout_raw/plantdoc_extract

echo "[stage5e1] 2/9 licence evidence grading + pool upstream audit + crossmatch reconciliation"
$PY tools/stage5e1_reconcile.py

echo "[stage5e1] 3/9 gates"
$PY tools/stage5e1_gates.py

echo "[stage5e1] 4/9 summary + reconciliation report"
$PY tools/stage5e1_final_report.py

echo "[stage5e1] 5/9 tests"
$PY -m pytest tests/ -q -rs > /tmp/stage5e1_pytest.txt 2>&1 || true
tail -n 20 /tmp/stage5e1_pytest.txt
$PY tools/stage5e1_record_tests.py

echo "[stage5e1] 6/9 regenerate summary/report with the recorded test result"
$PY tools/stage5e1_final_report.py

echo "[stage5e1] 7/9 re-run gates (report changes only)"
$PY tools/stage5e1_gates.py
$PY tools/stage5e1_final_report.py

echo "[stage5e1] done"
