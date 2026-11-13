#!/usr/bin/env bash
# usage: bash lib3d/audio/make_sfx.sh <scene.py> <SceneClass> <rendered_video.mp4>
cd "$(dirname "$0")/../.." || exit 1
FILE=$1; CLS=$2; VID=$3
if [ ! -f "$FILE" ] || [ -z "$CLS" ] || [ ! -f "$VID" ]; then
  echo "usage: bash lib3d/audio/make_sfx.sh <scene.py> <SceneClass> <rendered_video.mp4>"
  echo "check: scene=$FILE class=$CLS video=$VID"; exit 1
fi
MOD=$(echo "${FILE%.py}" | tr / .)
mkdir -p media/sfx
cat > scenes/_sfx_run.py <<PY
from manim import *
import importlib
from lib3d.audio.timeline import TimelineMixin
_m = importlib.import_module("$MOD")
class SfxRun(TimelineMixin, _m.$CLS):
    pass
PY
TL="media/sfx/${CLS}_timeline.json"; WAV="media/sfx/${CLS}.wav"
rm -f "$TL"
SFX_TIMELINE="$PWD/$TL" PYTHONPATH="$PWD" manim -r 480,270 --fps 30 -o SfxTimeline scenes/_sfx_run.py SfxRun > /tmp/sfx_tl.log 2>&1
if [ ! -f "$TL" ]; then echo "TIMELINE FAILED - first error:"; grep -m1 -B2 -A6 -E "Error|Traceback" /tmp/sfx_tl.log; exit 1; fi
python3 lib3d/audio/sfx.py "$TL" "$WAV" || exit 1
OUTV="${VID%.mp4}_sfx.mp4"
ffmpeg -y -loglevel error -i "$VID" -i "$WAV" -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest -movflags +faststart "$OUTV" || exit 1
VD=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$VID")
TD=$(python3 -c "import json;print(round(json.load(open('$TL'))['total'],2))")
echo "video duration: ${VD}s | timeline: ${TD}s (should match within ~0.1s)"
echo "output: $OUTV"
