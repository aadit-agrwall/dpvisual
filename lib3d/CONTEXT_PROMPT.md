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

lib3d/audio/sfx.py
  def T(d)
  def expd(t, tau)
  def noise(d)
  def lp(x, k)
  def hp(x, k=8)
  def atk(n, ms=2)
  def sweep(f0, f1, d)
  def hump(n, p=1.0)
  def pop(p=1.0, v=0.4)
  def ping(i, v=0.42)
  def untick(v=0.25)
  def tick(v=0.22, f=1500)
  def pluck(semi, v=0.3)
  def boing(v=0.3)
  def typing(d)
  def draw(d)
  def whoosh(d)
  def fade_out(d)
  def zap()
  def grow()
  def bonk()
  def chime()
  def build(data)
  def main()

lib3d/audio/timeline.py
  class TimelineMixin: play, move_camera, tear_down

lib3d/core/engine3d.py
  def box(size=(1, 1, 1), at=ORIGIN, sw=3)
  def cylinder(radius=0.6, height=1.6, at=ORIGIN, direction=OUT)
  def sphere(radius=0.7, at=ORIGIN)
  def cone(radius=0.6, height=1.4, at=ORIGIN, direction=OUT)
  def torus(major=1.0, minor=0.35, at=ORIGIN)
  def arrow3d(a, b, thickness=0.025, color=K)
  def segment(a, b, width=3, color=K, dashed=False)
  def axes3d(xr=(-3, 3, 1), yr=(-3, 3, 1), zr=(0, 3, 1), lengths=(6, 6, 3))
  def grid_floor(half=3, step=1, z=0.0, opacity=0.22)
  def curve3d(fn, t_range=(0, 1), color=K, width=5)
  def surface(fn, u_range=(-2.5, 2.5), v_range=(-2.5, 2.5), res=(22, 22))
  def bowl_surface(k=0.15, rng=2.5, res=(22, 22))
  def saddle_surface(k=0.18, rng=2.5, res=(22, 22))
  def gaussian_surface(mu=(0, 0), sigma=1.0, height=2.0, rng=3.0, res=(24, 24))
  def plane_poly(center=ORIGIN, normal=OUT, size=3.0, fill=0.25)
  def scatter_classes(n=12, seed=3, sep=2.0, spread=0.55, center=ORIGIN)
  def bars3d(grid, cell=0.5, gap=0.12, height_scale=1.0, at=ORIGIN)
  def matrix_grid(rows=4, cols=4, cell=0.5, at=ORIGIN, z=0.0)
  def grid_cell(grid, cols, i, j)
  def voxels(shape=(2, 2, 2), cell=0.5, gap=0.06, at=ORIGIN)
  def cnn_stack(n=4, w=1.6, h=1.6, depth=0.12, gap=0.35, at=ORIGIN)
  def neural_net(layers=(3, 5, 5, 2), gap_x=2.0, spread=1.0)
  class Net3D: reveal, fire_layer, forward_pass, reset, fade
  class Perceptron3D: reveal, fire, reset, fade
  class Stage3D: setup_stage, view, tour, orbit, dolly, spin, hud, caption, label3d, reveal, keep, clear_scene, blue, unblue, pulse, signal

lib3d/ml/basics.py
  class CapacityDemo3D: reveal, fit
  class LinearFit3D: reveal, fit, residuals
  class KFold3D: reveal, cycle
  class KNN3D: reveal, classify
  class PCA3D: reveal, show_axes, project, unproject
  class SGDLandscape: descend_noisy
  class MinibatchDemo: reveal, draw, sweep
  class MLRecipe3D: reveal, run, reset
  class CurseOfDim3D: reveal, sparse
  class ManifoldDemo: reveal, unroll, roll

lib3d/ml/basics_fit.py
  class CapacityDemo3D: reveal, fit
  class LinearFit3D: reveal, fit, residuals

lib3d/ml/classic.py
  def sigmoid_surface(w=(1.2, 0.8), b=0.0, rng=3.0, height=2.0, res=(24, 24))
  def boundary_line(w=(1.2, 0.8), b=0.0, rng=3.0, z=0.0, dashed=False, width=5)
  def blobs2d(n=12, seed=2, sep=2.4, spread=0.55, direction=(1, 1), z=0.06)
  class KMeans3D: reveal, step
  class SVMMargin3D: reveal, show_margin, highlight_support
  class DecisionTree3D: reveal, fire_path, reset
  def embedding_cloud(k=3, per=9, seed=5, spread=0.5, rng_=2.2, at=ORIGIN)
  def show_neighbors(scene, grp, pts, idx=0, k=3, run=1.0)
  def batch_stack(n=4, size=4, cell=0.32, gap_z=0.5, at=ORIGIN)

lib3d/ml/cnn.py
  class ConvSweep: reveal, sweep, reset
  class PoolDemo: reveal, pool
  class CNN3D: reveal, forward, reset

lib3d/ml/estimation.py
  class BiasVariance3D: reveal, fit, show_mean
  class MLE3D: reveal, ascend
  class Bayesian3D: reveal, update
  class KernelTrick3D: reveal, lift, unlift
  class Smoothness3D: reveal, show_bumpy, regularize

lib3d/ml/helpers.py
  def lit(m, op=0.85)
  def unlit(m)
  def tags(scene, items, size=24, run=0.4)
  def flashes(scene, pairs, run=0.6, width=5)

lib3d/ml/sequence.py
  class RNNUnrolled3D: reveal, run, reset
  class LSTMCell3D: reveal, fire, reset
  class Attention3D: reveal, focus, unfocus, sweep

lib3d/ml/training.py
  class CompGraph3D: linear_loss, reveal, forward, backward, reset
  def descent_path(fn, start, lr=0.1, steps=30, momentum=0.0, eps=0.0001)
  class LossLandscape: reveal, descend
  def gradient_arrow(fn, p, scale=1.5, eps=0.0001, descent=True)
  class Autoencoder3D: reveal, squeeze
  def dropout(scene, net, layer, idx, run=0.6)
  def undrop(scene, net, layer, idx, run=0.5)
  def backprop(scene, net, step=0.6)
  class ResBlock3D: reveal, fire, reset
