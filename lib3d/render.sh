#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
[ $# -ge 2 ] || { echo "usage: bash lib3d/render.sh <scene_file.py> <ClassName> [extra manim flags]"; exit 1; }
F="$1"; C="$2"; shift 2
source .venv/bin/activate 2>/dev/null
PYTHONPATH="$PWD" manim -qh -p "$F" "$C" "$@"
