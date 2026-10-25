#!/usr/bin/env bash
# Reproduce the Stage 5D corpus validation end to end, without interaction.
#
# Every step is deterministic.  Large artefacts stay outside Git:
#   data/corpus_v3/images/*.csv, R3v2_strict_rgb_{train,val}.csv,
#   data/corpus_v3/quarantine_reasons.csv,
#   data/corpus_v3/holdout_hash_index.csv
#
# The single GPU is addressed as cuda:1 (project rule: GPU 1 only).
#
#   STAGE5D_SKIP_EMBEDDINGS=1  build without the DINOv2 confirmation pass
#                              (faster, but the published manifest uses it)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${ROOT}/.venv-uv/bin/python"
export AGRI_DATA_ROOT="${AGRI_DATA_ROOT:-/data/datasets/Stage5C}"

cd "${ROOT}"

echo '== 1. preflight'
"${PY}" -I tools/stage5d_preflight.py --data-root "${AGRI_DATA_ROOT}"

echo '== 2. threshold trace'
"${PY}" -I tools/stage5d_audit_score_scales.py

echo '== 3. legacy funnel reconciliation'
"${PY}" -I tools/stage5d_reconcile_funnel.py

echo '== 4. repaired IQA (only if the derived ledger is absent)'
for source in agrifm_ref1 agrifm_ref57; do
  if [ ! -f "data/corpus_v3/derived_iqa/${source}.csv" ]; then
    "${PY}" -I tools/stage5d_compute_missing_iqa.py --source "${source}"
  fi
done

echo '== 5. build R3v2_strict_rgb'
if [ "${STAGE5D_SKIP_EMBEDDINGS:-0}" = "1" ]; then
  "${PY}" -I tools/stage5d_build_corpus.py --no-embeddings
else
  "${PY}" -I tools/stage5d_build_corpus.py
fi

echo '== 6. modality audit'
"${PY}" -I tools/stage5d_validate_modality.py

echo '== 7. synthetic QC calibration (>=200 base images)'
"${PY}" -I tools/stage5d_synthetic_qc.py --per-stratum 30 --strata 7

echo '== 8. threshold sensitivity'
"${PY}" -I tools/stage5d_sensitivity.py

echo '== 9. holdout cross-check'
"${PY}" -I tools/stage5d_holdout_audit.py

echo '== 10. publication materials (sources, dedup, groups, policy diff)'
"${PY}" -I tools/stage5d_publish_materials.py

echo '== 11. sampling profiles'
"${PY}" -I tools/stage5d_sampling_profiles.py

echo '== 12. tables and figures'
"${PY}" -I tools/stage5d_generate_tables.py

echo '== 13. gates and final report'
"${PY}" -I tools/stage5d_final_report.py

echo '== 14. loader smoke test (two batches, no optimizer)'
"${PY}" -I tools/stage5d_loader_smoke.py

echo '== 15. second independent build for the byte-identical gate'
"${PY}" -I tools/stage5d_rebuild_check.py

echo '== 16. tests'
"${PY}" -m pytest tests/ -q --no-header

echo 'done'
