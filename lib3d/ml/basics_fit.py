"""3D-aware fitting demos: data spread in depth (y), fits drawn as sheets, floor grid + axes."""
from manim import *
import numpy as np
from lib3d.core.engine3d import *
from lib3d.ml.helpers import *

YS = (-1.4, -0.7, 0.0, 0.7, 1.4)


def _stage_floor(scene, at, z=-3.3):
    fl = grid_floor(3, 1, z=z, opacity=0.18)
    fl.shift(at)
    scene.play(FadeIn(fl), run_time=0.5)
    scene.keep(fl)
    return fl


class CapacityDemo3D:
    """Noisy points scattered in depth. fit(scene, degree) draws the model as a curved sheet."""
    def __init__(self, n=11, seed=3, noise=0.25, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        self.x = np.linspace(-2.6, 2.6, n)
        self.y = 0.35 * self.x ** 2 - 0.8 + rng.normal(0, noise, n)
        self.d = rng.uniform(-1.3, 1.3, n)
        self.dots = VGroup(*[Dot3D(self.at + np.array([a, e, b]), radius=0.11, color=K)
                             for a, b, e in zip(self.x, self.y, self.d)])
        self.group = VGroup(self.dots)
        self.curves, self.tag = [], []

    def reveal(self, scene, run=1.0):
        scene.view("oblique", 1.0)
        _stage_floor(scene, self.at)
        scene.play(FadeIn(self.dots), run_time=run)
        scene.keep(self.group)

    def fit(self, scene, degree, run=1.0, color=ACCENT):
        p = np.poly1d(np.polyfit(self.x, self.y, degree))
        at = self.at
        sheet = VGroup(*[curve3d(lambda t, e=e: at + np.array([t, e, float(np.clip(p(t), -3.2, 3.2))]),
                                 (-2.8, 2.8), color, 4) for e in YS])
        names = {1: "underfit", 2: "good fit", 8: "overfit"}
        label = names.get(degree, f"degree {degree}")
        old = self.curves + self.tag
        if old:
            scene.play(*[FadeOut(m) for m in old], run_time=0.4)
        scene.play(*[Create(c) for c in sheet], run_time=run)
        scene.keep(sheet)
        scene.orbit(0.35, run=1.2)
        self.curves = [sheet]
        self.tag = tags(scene, [(label, at + np.array([0, 0, -3.0]))], 28)


class LinearFit3D:
    """Linear regression in 3D: the fitted line is a sheet; residuals are vertical drops."""
    def __init__(self, n=12, seed=2, noise=0.3, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        self.x = np.linspace(-2.6, 2.6, n)
        self.y = 0.8 * self.x + 0.2 + rng.normal(0, noise, n)
        self.d = rng.uniform(-1.3, 1.3, n)
        self.dots = VGroup(*[Dot3D(self.at + np.array([a, e, b]), radius=0.11, color=K)
                             for a, b, e in zip(self.x, self.y, self.d)])
        self.w, self.b = 0.0, -1.0
        self.sheet = self._sheet(self.w, self.b)
        self.group = VGroup(self.dots, self.sheet)

    def _ends(self, w, b, e):
        return (self.at + np.array([-3.0, e, -3.0 * w + b]), self.at + np.array([3.0, e, 3.0 * w + b]))

    def _sheet(self, w, b):
        return VGroup(*[Line(*self._ends(w, b, e), color=ACCENT, stroke_width=4) for e in YS])

    def reveal(self, scene, run=1.0):
        scene.view("oblique", 1.0)
        _stage_floor(scene, self.at)
        scene.play(FadeIn(self.dots), Create(self.sheet), run_time=run)
        scene.keep(self.group)

    def fit(self, scene, steps=8, lr=0.25, step_run=0.3):
        w, b = 0.0, -1.0
        for _ in range(steps):
            err = w * self.x + b - self.y
            w -= lr * float(np.mean(err * self.x)); b -= lr * float(np.mean(err))
            scene.play(*[ln.animate.put_start_and_end_on(*self._ends(w, b, e))
                         for ln, e in zip(self.sheet, YS)], run_time=step_run)
        self.w, self.b = w, b
        scene.orbit(0.4, run=1.4)

    def residuals(self, scene, run=0.8):
        segs = VGroup(*[Line(self.at + np.array([a, e, c]), self.at + np.array([a, e, self.w * a + self.b]),
                             color=ACCENT, stroke_width=3)
                        for a, c, e in zip(self.x, self.y, self.d)])
        scene.play(Create(segs), run_time=run)
        scene.keep(segs)
        return segs
