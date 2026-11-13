"""Chapter-5 style ML basics: capacity, regression, PCA, kNN, SGD, curse of dimensionality, manifolds."""
from manim import *
import numpy as np
from lib3d.core.engine3d import *
from lib3d.ml.helpers import *
from lib3d.ml.helpers import _u
from lib3d.ml.training import LossLandscape, descent_path


# ------------------------------------------------------------ fitting
class CapacityDemo3D:
    """Points in the x-z plane (use view('front')). fit(scene, degree) draws a polynomial fit."""
    def __init__(self, n=9, seed=3, noise=0.25, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        self.x = np.linspace(-2.6, 2.6, n)
        self.y = 0.35 * self.x ** 2 - 0.8 + rng.normal(0, noise, n)
        self.dots = VGroup(*[Dot3D(self.at + np.array([a, 0, b]), radius=0.1, color=K)
                             for a, b in zip(self.x, self.y)])
        self.group = VGroup(self.dots)
        self.curve = None
        self.tag = []

    def reveal(self, scene, run=1.0):
        scene.play(FadeIn(self.dots), run_time=run)
        scene.keep(self.group)

    def fit(self, scene, degree, run=1.0, color=ACCENT):
        p = np.poly1d(np.polyfit(self.x, self.y, degree))
        at = self.at
        new = curve3d(lambda t: at + np.array([t, 0.0, float(np.clip(p(t), -3.2, 3.2))]),
                      (-2.8, 2.8), color, 5)
        names = {1: "underfit", 2: "good fit", 8: "overfit"}
        label = names.get(degree, f"degree {degree}")
        old = ([self.curve] if self.curve is not None else []) + self.tag
        if old:
            scene.play(*[FadeOut(m) for m in old], run_time=0.4)
        scene.play(Create(new), run_time=run)
        scene.keep(new)
        self.curve = new
        self.tag = tags(scene, [(label, at + np.array([0, 0, -2.6]))], 28)


class LinearFit3D:
    """Linear regression in the x-z plane (view 'front'): line moves to fit, then residuals."""
    def __init__(self, n=12, seed=2, noise=0.3, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        self.x = np.linspace(-2.6, 2.6, n)
        self.y = 0.8 * self.x + 0.2 + rng.normal(0, noise, n)
        self.dots = VGroup(*[Dot3D(self.at + np.array([a, 0, b]), radius=0.1, color=K)
                             for a, b in zip(self.x, self.y)])
        self.line = self._line(0.0, -1.0)
        self.group = VGroup(self.dots, self.line)

    def _line(self, w, b, color=ACCENT):
        a = self.at + np.array([-3.0, 0, -3.0 * w + b])
        c = self.at + np.array([3.0, 0, 3.0 * w + b])
        return Line(a, c, color=color, stroke_width=5)

    def reveal(self, scene, run=1.0):
        scene.play(FadeIn(self.dots), Create(self.line), run_time=run)
        scene.keep(self.group)

    def fit(self, scene, steps=8, lr=0.25, step_run=0.3):
        w, b = 0.0, -1.0
        for _ in range(steps):
            err = w * self.x + b - self.y
            w -= lr * float(np.mean(err * self.x)); b -= lr * float(np.mean(err))
            t = self._line(w, b)
            scene.play(self.line.animate.put_start_and_end_on(*t.get_start_and_end()), run_time=step_run)
        self.w, self.b = w, b

    def residuals(self, scene, run=0.8):
        w, b = getattr(self, "w", 0.8), getattr(self, "b", 0.2)
        segs = VGroup(*[Line(self.at + np.array([a, 0, c]), self.at + np.array([a, 0, w * a + b]),
                             color=K, stroke_width=2, stroke_opacity=0.7)
                        for a, c in zip(self.x, self.y)])
        scene.play(Create(segs), run_time=run)
        scene.keep(segs)
        return segs


class KFold3D:
    def __init__(self, folds=5, cell=0.8, at=ORIGIN):
        self.folds = folds
        self.at = np.array(at, dtype=float)
        self.grid = matrix_grid(1, folds, cell, self.at)
        self.group = VGroup(self.grid)

    def reveal(self, scene, run=1.0):
        scene.play(FadeIn(self.grid), run_time=run)
        scene.keep(self.group)
        tags(scene, [("blue = held-out fold", self.at + np.array([0, -1.0, 0]))], 26)

    def cycle(self, scene, step=0.6):
        for k in range(self.folds):
            scene.play(*[unlit(grid_cell(self.grid, self.folds, 0, j)) for j in range(self.folds)],
                       lit(grid_cell(self.grid, self.folds, 0, k)), run_time=step)


# ----------------------------------------------------- supervised / unsup.
class KNN3D:
    def __init__(self, n=8, seed=3, sep=2.6, spread=0.55, at=ORIGIN):
        at = np.array(at, dtype=float)
        self.pos, self.neg, pp, nn = scatter_classes(n, seed, sep, spread, at)
        self.mobs = list(self.pos) + list(self.neg)
        self.pts = [np.array(p, dtype=float) for p in list(pp) + list(nn)]
        self.npos = len(pp)
        self.q = at + np.array([0.3, 0.2, 0.2])
        self.qdot = sphere(0.16, self.q)
        self.group = VGroup(self.pos, self.neg, self.qdot)

    def reveal(self, scene, run=1.2):
        scene.play(FadeIn(self.pos), FadeIn(self.neg), FadeIn(self.qdot), run_time=run)
        scene.keep(self.group)

    def classify(self, scene, k=3, run=1.0):
        d = [float(np.linalg.norm(p - self.q)) for p in self.pts]
        idx = [int(j) for j in np.argsort(d)[:k]]
        lines = VGroup(*[Line(self.q, self.pts[j], color=ACCENT, stroke_width=3) for j in idx])
        scene.play(*[lit(self.mobs[j]) for j in idx], lit(self.qdot), *[Create(l) for l in lines],
                   run_time=run)
        scene.keep(lines)
        npos = sum(1 for j in idx if j < self.npos)
        tags(scene, [(f"votes: {npos} vs {k - npos}", self.q + np.array([0, 0, 1.1]))], 26)
        return idx


class PCA3D:
    def __init__(self, n=40, seed=5, scales=(2.0, 0.9, 0.25), at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        X = (rng.normal(0, 1, (n, 3)) * np.array(scales)) @ Q.T
        X = X - X.mean(0)
        _, S, Vt = np.linalg.svd(X, full_matrices=False)
        self.X, self.V, self.sd = X, Vt, S / np.sqrt(n)
        self.dots = VGroup(*[Dot3D(self.at + x, radius=0.08, color=K) for x in X])
        self.group = VGroup(self.dots)

    def reveal(self, scene, run=1.2):
        scene.play(FadeIn(self.dots), run_time=run)
        scene.keep(self.group)

    def show_axes(self, scene, run=1.0):
        arrows, items = [], []
        for i in range(3):
            tip = self.at + self.V[i] * (1.0 + 1.2 * self.sd[i])
            arrows.append(arrow3d(self.at, tip, color=ACCENT))
            items.append((f"PC{i + 1}", tip + self.V[i] * 0.35))
        scene.play(*[FadeIn(a) for a in arrows], run_time=run)
        scene.keep(*arrows)
        tags(scene, items, 24)

    def _proj(self, k):
        V = self.V[:k]
        return self.at + (self.X @ V.T) @ V

    def project(self, scene, k=2, run=1.4):
        P = self._proj(k)
        scene.play(*[d.animate.move_to(p) for d, p in zip(self.dots, P)], run_time=run)

    def unproject(self, scene, run=1.0):
        scene.play(*[d.animate.move_to(self.at + x) for d, x in zip(self.dots, self.X)], run_time=run)


# ---------------------------------------------------------------- SGD
class SGDLandscape(LossLandscape):
    """LossLandscape + noisy (stochastic) descent. Inherited descend() = full-batch."""
    def descend_noisy(self, scene, start=(2.2, 1.8), lr=1.2, steps=12, noise=0.4,
                      seed=0, step_run=0.25, eps=1e-4):
        rng = np.random.default_rng(seed)
        f = self.fn
        x = np.array(start, dtype=float)
        pts = [x.copy()]
        for _ in range(steps):
            g = np.array([(f(x[0] + eps, x[1]) - f(x[0] - eps, x[1])) / (2 * eps),
                          (f(x[0], x[1] + eps) - f(x[0], x[1] - eps)) / (2 * eps)])
            x = np.clip(x - lr * (g + rng.normal(0, noise, 2)), -2.4, 2.4)
            pts.append(x.copy())
        P = [np.array([p[0], p[1], f(p[0], p[1]) + 0.07]) for p in pts]
        dot = Dot3D(P[0], radius=0.14, color=ACCENT)
        scene.play(FadeIn(dot), run_time=0.4)
        trail = VGroup()
        for a, b in zip(P[:-1], P[1:]):
            seg = segment(a, b, width=4, color=ACCENT)
            scene.play(dot.animate.move_to(b), Create(seg), run_time=step_run)
            trail.add(seg)
        scene.keep(dot, trail)
        return dot, trail, P


class MinibatchDemo:
    def __init__(self, rows=6, cols=10, cell=0.38, seed=1, at=ORIGIN):
        self.rows, self.cols = rows, cols
        self.at = np.array(at, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.grid = matrix_grid(rows, cols, cell, self.at)
        self.cell = cell
        self.group = VGroup(self.grid)

    def reveal(self, scene, run=1.0):
        scene.play(FadeIn(self.grid), run_time=run)
        scene.keep(self.group)
        tags(scene, [("training set", self.at + np.array([0, -(self.rows * self.cell / 2 + 0.5), 0]))], 26)

    def draw(self, scene, b=8, hold=0.5, run=0.4):
        idx = self.rng.choice(self.rows * self.cols, size=b, replace=False)
        cells = [grid_cell(self.grid, self.cols, int(k) // self.cols, int(k) % self.cols) for k in idx]
        scene.play(*[lit(c) for c in cells], run_time=run)
        scene.wait(hold)
        scene.play(*[unlit(c) for c in cells], run_time=run)

    def sweep(self, scene, batches=3, b=8):
        for _ in range(batches):
            self.draw(scene, b)


# -------------------------------------------------- build an ML algorithm
class MLRecipe3D:
    """dataset -> model -> cost -> optimizer, with an update loop back to the model."""
    def __init__(self, gap=3.0, at=ORIGIN):
        at = np.array(at, dtype=float)
        self.P = [at + np.array([(k - 1.5) * gap, 0, 0]) for k in range(4)]
        self.boxes = VGroup(*[box((0.9, 0.9, 0.9), p) for p in self.P])
        h = np.array([0.45, 0, 0])
        self.segs = [(self.P[k] + h, self.P[k + 1] - h) for k in range(3)]
        self.arrows = VGroup(*[Line(a, b, color=K, stroke_width=3) for a, b in self.segs])
        d = np.array([0, 0, -1.6])
        self.loop = [self.P[3] + np.array([0, 0, -0.45]), self.P[3] + d, self.P[1] + d,
                     self.P[1] + np.array([0, 0, -0.45])]
        self.loop_mob = VMobject(color=K, stroke_width=3).set_points_as_corners(self.loop)
        self.group = VGroup(self.boxes, self.arrows, self.loop_mob)

    def reveal(self, scene, run=1.6):
        scene.play(LaggedStart(*[FadeIn(b) for b in self.boxes], lag_ratio=0.3), run_time=run * 0.6)
        scene.play(Create(self.arrows), Create(self.loop_mob), run_time=run * 0.4)
        scene.keep(self.group)
        names = ["data $X, y$", r"model $f(x;\theta)$", r"cost $J(\theta)$", "optimizer"]
        tags(scene, [(n, p + np.array([0, 0, 0.95])) for n, p in zip(names, self.P)], 24)

    def run(self, scene, step=0.5):
        scene.play(lit(self.boxes[0]), run_time=step * 0.4)
        for k in range(3):
            flashes(scene, [self.segs[k]], run=step)
            scene.play(lit(self.boxes[k + 1]), run_time=step * 0.4)
        scene.play(ShowPassingFlash(self.loop_mob.copy().set_stroke(ACCENT, width=6), time_width=0.6),
                   run_time=step * 2)

    def reset(self, scene, run=0.5):
        scene.play(*[unlit(b) for b in self.boxes], run_time=run)


# ------------------------------------------- challenges motivating DL
class CurseOfDim3D:
    """4 bins per axis: 1D=4 cells, 2D=16, 3D=64. sparse() shows few samples in many cells."""
    def __init__(self, bins=4, s=0.45, seed=7, at=ORIGIN):
        at = np.array(at, dtype=float)
        self.bins, self.rng = bins, np.random.default_rng(seed)
        c = (bins - 1) / 2
        self.centers = [at + np.array([-4.5, 0, 0]), at + np.array([-0.8, 0, 0]), at + np.array([3.0, 0, 0])]
        mk = lambda p: box((s * 0.65,) * 3, p, sw=1.5)
        self.g1 = VGroup(*[mk(self.centers[0] + np.array([(i - c) * s, 0, 0])) for i in range(bins)])
        self.g2 = VGroup(*[mk(self.centers[1] + np.array([(i - c) * s, 0, (j - c) * s]))
                           for i in range(bins) for j in range(bins)])
        self.g3 = VGroup(*[mk(self.centers[2] + np.array([(i - c) * s, (j - c) * s, (k - c) * s]))
                           for i in range(bins) for j in range(bins) for k in range(bins)])
        self.groups = [self.g1, self.g2, self.g3]
        self.s = s
        self.group = VGroup(*self.groups)

    def reveal(self, scene, dims=(1, 2, 3), run=0.9):
        for d in dims:
            g = self.groups[d - 1]
            scene.play(LaggedStart(*[FadeIn(m) for m in g], lag_ratio=0.02), run_time=run)
            scene.keep(g)
            tags(scene, [(f"${self.bins}^{d} = {self.bins ** d}$ cells",
                          self.centers[d - 1] + np.array([0, 0, -(self.bins * self.s / 2 + 0.7)]))], 26)

    def sparse(self, scene, n=3, run=0.7):
        sel = []
        for g in self.groups:
            idx = self.rng.choice(len(g), size=min(n, len(g)), replace=False)
            sel += [g[int(i)] for i in idx]
        scene.play(*[lit(m) for m in sel], run_time=run)


class ManifoldDemo:
    """A spiral curve (1-D manifold) in 3-D; unroll() maps its points onto a straight line."""
    def __init__(self, n=28, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        a = self.at
        self.f = lambda t: a + np.array([(0.5 + 0.2 * t) * np.cos(t), (0.5 + 0.2 * t) * np.sin(t), 0.2 * t - 1.0])
        T = 3 * np.pi
        dense = np.linspace(0, T, 2000)
        P = np.array([self.f(t) for t in dense])
        cum = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
        ts = np.linspace(0.3, T - 0.2, n)
        s = np.interp(ts, dense, cum) * 0.45
        self.flat = [a + np.array([v - s.mean(), 0, 0]) for v in s]
        self.curve = curve3d(self.f, (0, T), K, 3)
        self.dots = VGroup(*[Dot3D(self.f(t), radius=0.09, color=K) for t in ts])
        self.ts = ts
        self.group = VGroup(self.curve, self.dots)
        self.line = None

    def reveal(self, scene, run=1.4):
        scene.play(Create(self.curve), FadeIn(self.dots), run_time=run)
        scene.keep(self.group)

    def unroll(self, scene, run=1.8):
        ends = (self.flat[0] + np.array([-0.3, 0, 0]), self.flat[-1] + np.array([0.3, 0, 0]))
        self.line = Line(*ends, color=K, stroke_width=3)
        scene.play(FadeOut(self.curve), *[d.animate.move_to(p) for d, p in zip(self.dots, self.flat)],
                   Create(self.line), run_time=run)
        scene.keep(self.line)

    def roll(self, scene, run=1.6):
        scene.play(FadeOut(self.line), *[d.animate.move_to(self.f(t)) for d, t in zip(self.dots, self.ts)],
                   FadeIn(self.curve), run_time=run)
