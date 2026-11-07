#!/bin/bash
# Stage 8 Phase 4: execute all frozen formal runs, serial on GPU 1,
# task by task.  The runner skips already-complete rows (resume-safe),
# so this script can be re-launched after a crash.
set -u
cd /fj/jepa-二维图像任务
PY=./.venv-uv/bin/python

for task in phenobench_sem plantdoc_cls deepweeds_cls cropandweed_sem gwhd_det; do
  echo "########## TASK $task primary ##########"
  $PY tools/stage8_phase4_runner.py --tasks "$task" --roles frozen \
      || { echo "!!! TASK $task failed (see log)"; exit 1; }
  echo "########## TASK $task exploratory ##########"
  $PY tools/stage8_phase4_runner.py --tasks "$task" --roles exploratory \
      || { echo "!!! TASK $task exploratory failed"; exit 1; }
done
echo "ALL_PHASE4_DONE"