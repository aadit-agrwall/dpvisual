#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
export PYTHONPATH="$PWD"
case "${1:-ml}" in
  ml)   F=lib3d/tests/test_3d_ml.py;      C=Test3DML;    R=/tmp/engine3d_ml_test.txt;   D=test_3d_ml ;;
  ch05) F=lib3d/tests/test_ch05.py;       C=Test3DCh05;  R=/tmp/engine3d_ch05_test.txt; D=test_ch05 ;;
  est)  F=lib3d/tests/test_estimation.py; C=Test3DEst;   R=/tmp/engine3d_est_test.txt;  D=test_estimation ;;
  *) echo "usage: run_tests.sh [ml|ch05|est]"; exit 1 ;;
esac
rm -f "$R"
python3 -c "import lib3d; print('import OK, lib3d', lib3d.__version__)" || exit 1
manim -ql "$F" "$C" > /tmp/lib3d_test.log 2>&1; RC=$?
tail -n 4 /tmp/lib3d_test.log; echo "manim exit code: $RC"
echo "=== $C REPORT ==="; cat "$R" 2>/dev/null
if [ $RC -ne 0 ]; then echo "FAILED - see first error in /tmp/lib3d_test.log"; exit $RC; fi
SRC=$(find media/videos/$D -name "$C.mp4" -not -path '*partial_movie_files*' -type f -print0 | xargs -0 ls -t | head -1)
open "$SRC"
