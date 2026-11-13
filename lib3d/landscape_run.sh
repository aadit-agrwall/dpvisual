#!/usr/bin/env bash
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null
FPS="${FPS:-30}"; export FPS
WANT="145 146 147 148 149 150"
mkdir -p _land assets/bookpage assets/user-data/output media/sfx
grep -qx "_land/" .gitignore 2>/dev/null || echo "_land/" >> .gitignore

OK=""
for N in $WANT; do
  IMG=assets/bookpage/page_$N.png; SC=scenes/chapter05/page_$N.py
  if [ ! -f "$IMG" ]; then
    F=$(find ~/Downloads ~/Desktop ~/Documents/dpvisual -maxdepth 4 -name "page_$N.png" -not -path "*/_land/*" 2>/dev/null | head -n 1)
    if [ -n "$F" ]; then cp "$F" "$IMG"; echo "copied screenshot $F"; fi
  fi
  if [ -f "$IMG" ] && [ -f "$SC" ]; then OK="$OK $N"
  else echo "REPORT: MISSING page $N (screenshot: $([ -f $IMG ] && echo yes || echo NO), scene file: $([ -f $SC ] && echo yes || echo NO))"; fi
done
echo "REPORT: pages to build:$OK  (fps $FPS)"

for N in $OK; do
  if [ -f _land/content_$N.mp4 ]; then echo "REPORT: page $N already rendered, skipping"; continue; fi
  echo "rendering page $N at 1440x1440 ... (log /tmp/land_$N.log)"
  rm -f media/sfx/Page${N}_timeline.json
  SFX_TIMELINE="$PWD/media/sfx/Page${N}_timeline.json" PYTHONPATH="$PWD" \
    python3 -m manim -r 1440,1440 --fps $FPS --disable_caching -o content_$N scenes/chapter05/page_$N.py Page$N > /tmp/land_$N.log 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "REPORT: FAIL page $N render (exit $RC). last 30 lines:"; tail -n 30 /tmp/land_$N.log; continue
  fi
  V=$(find media/videos -name "content_$N.mp4" -not -path "*partial_movie_files*" -print0 | xargs -0 ls -t 2>/dev/null | head -n 1)
  if [ -n "$V" ] && [ -f media/sfx/Page${N}_timeline.json ]; then
    cp "$V" _land/content_$N.mp4; echo "REPORT: OK page $N rendered ($V)"
  else
    echo "REPORT: FAIL page $N output or timeline not found"
  fi
done

READY=""
for N in $OK; do [ -f _land/content_$N.mp4 ] && READY="$READY $N"; done
echo "REPORT: building video from:$READY"
if [ -n "$READY" ]; then
  python3 lib3d/landscape_build.py $READY
  F=$(ls -t assets/user-data/output/pages_*_landscape_2k.mp4 2>/dev/null | head -n 1)
  if [ -n "$F" ]; then open "$F"; fi
fi
