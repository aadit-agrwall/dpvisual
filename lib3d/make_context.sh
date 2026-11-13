#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
OUT=lib3d/CONTEXT_PROMPT.md
PY=.venv/bin/python; [ -x "$PY" ] || PY=python3

cat > "$OUT" <<'EOF'
# PROJECT CONTEXT (read fully before answering)

I am rendering the book "Deep Learning" (Goodfellow, Bengio, Courville) page by page as Manim animations. Chapter 5 (ML basics) is done. I reuse one shared 3D library across chapters.

## Environment
- macOS, zsh, project root ~/Documents/dpvisual, Python 3.12 in .venv, manim v0.20.1.
- Always run from the project root with PYTHONPATH="$PWD". Videos open with `open`.
- zsh quirks: run `setopt interactivecomments` before pasting commented blocks. Use `sed -i ''` (macOS form).

## How I want you to work
- Give me complete terminal blocks I can paste: cd to the project root first, back up before changing files, safe to re-run, no manual editing.
- Every new module comes with a smoke test and a printed REPORT of OK/FAIL lines. I paste the report back to you.
- If you need information (existing files, signatures, errors), give me a terminal command that prints it. Do not guess.
- Never assume the content of a file you have not seen. Ask me to cat it.

## The library: lib3d/ (chapter-independent, never put chapter code in it)
Usage in ANY chapter or page:  from manim import *  then  from lib3d import *

Layout:
- lib3d/core/theme.py: palette (K black, PAPER white, ACCENT blue #1F4FFF) and VIEWS camera dict.
- lib3d/core/engine3d.py: primitives and Stage3D.
- lib3d/core/guard.py: maps edge-on views to "slant" automatically.
- lib3d/ml/: helpers.py, cnn.py, sequence.py, training.py, classic.py, basics.py, basics_fit.py, estimation.py. Each file is registered with one line in lib3d/ml/__init__.py.
- lib3d/tests/: test_3d_ml.py, test_ch05.py, test_estimation.py. Run with: bash lib3d/run_tests.sh [ml|ch05|est]
- lib3d/lint_views.sh lists edge-on camera calls. lib3d/README.md is the cheat sheet.
- common/engine3d.py and common/engine3d_ml.py are thin legacy shims (importlib re-exports of lib3d). New code must use lib3d directly.
- A live list of classes, methods and functions is appended at the bottom of this prompt.

## Visual style (strict)
- Black and white only: black outlines, white fill, white background.
- ACCENT blue is the ONLY extra color and means "change" (firing, selection, active). Use lit(m) / unlit(m); never invent other colors.

## Stage3D API (every scene subclasses it)
- setup_stage(view="iso", zoom=0.85): call first in construct().
- view(name, run, zoom), tour(names), orbit(angle, run), dolly(zoom), spin(run, rate)
- caption(text): top title that auto-replaces the previous one. hud(text), label3d(text, pos)
- reveal(*mobs, anim="fade"|"create"|"grow", run): shows and registers objects.
- keep(*mobs): registers objects so clear_scene() removes them.
- clear_scene(run): fades out everything registered plus the caption.
- blue/unblue/pulse(mob), signal(a, b): blue emphasis helpers.
Primitives: box, sphere, cylinder, cone, torus, arrow3d, segment, axes3d, grid_floor, curve3d, surface, bowl_surface, saddle_surface, gaussian_surface, plane_poly, scatter_classes, bars3d, matrix_grid, grid_cell, voxels, cnn_stack, neural_net, Net3D, Perceptron3D.

## Component convention (follow it for every new component)
- A class with: __init__(..., at=ORIGIN) building mobjects (no animation); reveal(scene) that plays the intro, calls scene.keep(self.group) and adds labels with tags(); one or more action methods such as fire/run/step/fit/update; and reset(scene) where it makes sense.
- Helpers: lit/unlit (blue on/off), tags(scene, [(text, pos)]) for flat labels readable from any camera, flashes(scene, [(a, b)]) for signal flashes.
- A new topic gets its own file lib3d/ml/<topic>.py with the same header (manim, numpy, lib3d.core.engine3d, lib3d.ml.helpers) plus one registration line in lib3d/ml/__init__.py, a test in lib3d/tests/, and a run_tests.sh case.

## CAMERA RULE (a mistake already made once, never repeat)
- "front" and "side" look edge-on: anything laid out along x collapses into a flat line and viewers cannot tell what is happening. Never design scenes that need them.
- Use "slant" (left-to-right diagrams with visible depth), "oblique" (point clouds, surfaces, general 3D), "iso" (default), "top" ONLY for genuinely 2D grids (conv, pooling, minibatch).
- Every scene must show real 3D: spread data in depth (y), use a floor grid (grid_floor), sheets of parallel curves instead of a single line, and a short scene.orbit() after important reveals. Components set their own camera in reveal().
- guard.py auto-maps front/side to slant (override only with force=True). Run bash lib3d/lint_views.sh to find offenders.

## Known pitfalls
- manim exports a module named `core`, which shadows lib3d.core. Load by importlib.import_module("lib3d.core...") when needed. Do not use `import lib3d.core.x as y`.
- Do not put chapter-specific constants into lib3d. Chapter styling stays in the chapter files.

## Page rendering (how pages use the library)
- Each book page is one Manim scene class inheriting Stage3D, in scenes/ (existing examples: scenes/page001.py, scenes/page014.py, chapter folders scenes/chapter01..05). Render with:
    bash lib3d/render.sh scenes/<path>/<file>.py <ClassName>
  which runs: PYTHONPATH="$PWD" manim -qh -p scenes/<path>/<file>.py <ClassName>
- RENDER RULE: always render ONCE, directly at 1080p (-qh). Never propose a 480p/-ql test render first and then a second final render. Add --fps 30 only if I ask for faster renders.
- Page scene pattern:
    from manim import *
    from lib3d import *
    class Page123(Stage3D):
        def construct(self):
            self.setup_stage("iso")
            self.caption("Short title from the page")
            c = SomeComponent(); c.reveal(self); c.action(self); self.clear_scene()
  One caption per idea, narration order follows the book text, finish each idea with clear_scene().
- Audio/music/mux scripts live in common/ (music_*.py, mux_audio.sh, page_audio_loud.sh, append_page_image.sh, rename_bookpages.sh). I have not described how they work here. Ask me to print them before touching the audio or page-export pipeline.
- Before writing page scenes, ask me for: the page text or topic, and one existing finished page file (cat it) so you match my pacing and conventions.

## Status
- Chapter 5 library is complete and tested (conv, pooling, CNN, RNN, LSTM, attention, computation graph, loss landscape, SGD, autoencoder, ResBlock, k-means, SVM, tree, kNN, PCA, capacity, regression, k-fold, curse of dimensionality, manifold, bias/variance, MLE, Bayesian, kernel trick, smoothness).
- Planned next: Chapter 6 (Deep Feedforward Networks: XOR, activations, universal approximation, backprop graphs), then Transformer block, GAN, RL loop, Diffusion.

## Live API (auto-generated from the current code)
EOF

"$PY" - <<'PY' >> "$OUT"
import ast, pathlib
for f in sorted(pathlib.Path("lib3d").rglob("*.py")):
    if "tests" in f.parts or f.name == "__init__.py":
        continue
    t = ast.parse(f.read_text())
    items = []
    for n in t.body:
        if isinstance(n, ast.ClassDef):
            ms = [m.name for m in n.body if isinstance(m, ast.FunctionDef) and not m.name.startswith("_")]
            items.append(f"  class {n.name}: {', '.join(ms)}")
        elif isinstance(n, ast.FunctionDef) and not n.name.startswith("_"):
            items.append(f"  def {n.name}({ast.unparse(n.args)})")
    if items:
        print(f"\n{f}")
        print("\n".join(items))
PY
echo "wrote $OUT ($(wc -l < "$OUT") lines)"
