#!/usr/bin/env bash
# lists edge-on camera calls (guard auto-fixes them, but better to remove them)
cd "$(dirname "$0")/.." || exit 1
grep -rnE 'view\("(front|side)"|setup_stage\("(front|side)"' scenes lib3d --include=*.py \
  | grep -v "force=True" | grep -v "core/guard.py" | grep -v "tests/test_estimation.py" \
  || echo "lint OK: no edge-on views"
