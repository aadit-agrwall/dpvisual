#!/usr/bin/env bash
# usage: bash lib3d/landscape_run2.sh 141 142 143 144 145     (FPS=60 for 60 fps)
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null
FPS="${FPS:-30}"; export FPS
WANT="${*:-141 142 143 144 145}"
mkdir -p _land assets/bookpage assets/user-data/output media/sfx
grep -qx "_land/" .gitignore 2>/dev/null || echo "_land/" >> .gitignore

READY=""
for N in $WANT; do
  IMG=assets/bookpage/page_$N.png
  if [ ! -f "$IMG" ]; then echo "REPORT: MISSING screenshot $IMG"; continue; fi
  if [ -f _land/content_$N.mp4 ]; then echo "REPORT: page $N already rendered, skipping render"; READY="$READY $N"; continue; fi
  SC=$(grep -rlE "^class Page_?$N[^0-9A-Za-z_]" scenes --include=*.py | head -n 1)
  if [ -z "$SC" ]; then echo "REPORT: MISSING scene file for page $N (no 'class Page$N' under scenes/)"; continue; fi
  CLS=$(grep -hE "^class Page_?$N[^0-9A-Za-z_]" "$SC" | head -n 1 | sed -E 's/^class ([A-Za-z_0-9]+).*/\1/')
  echo "page $N: $SC  class $CLS  -> rendering 1440x1440 (log /tmp/land_$N.log)"
  rm -f media/sfx/Page${N}_timeline.json
  SFX_TIMELINE="$PWD/media/sfx/Page${N}_timeline.json" PYTHONPATH="$PWD:$PWD/common" \
    python3 -m manim -r 1440,1440 --fps $FPS --disable_caching -o content_$N "$SC" "$CLS" > /tmp/land_$N.log 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then echo "REPORT: FAIL page $N render (exit $RC). last 30 lines:"; tail -n 30 /tmp/land_$N.log; continue; fi
  V=$(find media/videos -name "content_$N.mp4" -not -path "*partial_movie_files*" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)
  if [ -n "$V" ]; then cp "$V" _land/content_$N.mp4; echo "REPORT: OK page $N rendered ($V)"; READY="$READY $N"
  else echo "REPORT: FAIL page $N output not found"; fi
done

echo "REPORT: building video from:$READY"
if [ -n "$READY" ]; then
  python3 lib3d/landscape_build.py $READY
  set -- $READY
  F="assets/user-data/output/pages_${1}_${@: -1}_landscape_2k.mp4"
  if [ -f "$F" ]; then echo "REPORT: OK saved $F"; open "$F"; fi
fi
