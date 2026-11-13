from manim import *
import numpy as np
from lib3d import *
from lib3d.ml.basics_fit import _stage_floor


def _bars(scene, at, vals, names, size=0.8, gap=1.6, run=1.0):
    """3 labelled bars (own boxes, so layout is exact). Returns the bar mobjects."""
    at = np.array(at, dtype=float)
    L = len(vals)
    bars, items = [], []
    for k, (h, n) in enumerate(zip(vals, names)):
        x = (k - (L - 1) / 2) * gap
        bars.append(box((size, size, h), at + np.array([x, 0, h / 2])))
        items.append((n, at + np.array([x, 0, -0.55])))
    scene.play(LaggedStart(*[FadeIn(b) for b in bars], lag_ratio=0.25), run_time=run)
    scene.keep(*bars)
    tags(scene, items, 24)
    return bars


class Page141(Stage3D):
    def construct(self):
        self.setup_stage("oblique")

        # 1. kernels struggle -> deep learning renaissance (Hinton 2006, RBF SVM on MNIST)
        self.caption("Generic kernels struggle to generalize")
        s = SVMMargin3D(); s.reveal(self); s.show_margin(self); s.highlight_support(self)
        self.orbit(0.5, run=2.0)
        tags(self, [("RBF-kernel SVM on MNIST", ORIGIN + np.array([0, 0, 3.0]))], 26)
        self.wait(0.5)
        self.clear_scene()

        # 2. k-nearest neighbors: no training stage, look at the neighbors at test time
        self.caption("5.7.3  k-nearest neighbors: nonparametric")
        k = KNN3D(); k.reveal(self)
        tags(self, [("no training stage", np.array([0, -2.8, -1.8]))], 26)
        k.classify(self, 3)
        self.orbit(0.5, run=2.0)
        self.clear_scene()

        # 3. average the neighbors' one-hot codes -> probability distribution
        self.caption("Average of one-hot codes = class probabilities")
        self.view("oblique", 1.0)
        _stage_floor(self, ORIGIN, z=-0.05)
        bars = _bars(self, ORIGIN, [2.0, 1.0, 0.3], ["class 1", "class 2", "class 3"])
        self.play(lit(bars[0]), run_time=0.5)
        tags(self, [("neighbors vote: 2/3, 1/3, 0", np.array([0, 0, 2.9]))], 24)
        self.orbit(0.5, run=2.0)
        self.clear_scene()

        # 4. error: 1-NN -> 2x Bayes error; voting over all neighbors -> Bayes error
        self.caption("High capacity: 1-NN gives 2x Bayes error, voting gives Bayes error")
        self.view("oblique", 1.0)
        _stage_floor(self, ORIGIN, z=-0.05)
        bars = _bars(self, ORIGIN, [1.0, 2.0, 1.0], ["Bayes error", "1-NN", "all-neighbor vote"], gap=2.0)
        self.play(lit(bars[1]), run_time=0.6)
        tags(self, [("infinite data", np.array([0, 0, 2.9]))], 26)
        self.wait(0.4)
        self.play(unlit(bars[1]), lit(bars[2]), run_time=0.6)
        self.orbit(0.5, run=2.0)
        self.clear_scene()

        # 5. weakness: cannot learn that one feature matters more
        self.caption(r"Weakness: only $x_1$ is relevant, $y = x_1$")
        self.view("oblique", 1.0)
        rng = np.random.default_rng(11)
        P = rng.uniform(-2.4, 2.4, (26, 3))          # columns: x1, x2, x3
        q = np.array([0.0, 0.0, 0.0])
        d = np.linalg.norm(P - q, axis=1)
        near = [int(i) for i in np.argsort(d)[:3]]    # what k-NN picks (full distance)
        true = [int(i) for i in np.argsort(np.abs(P[:, 0]))[:3]]  # what it should use (x1 only)
        _stage_floor(self, ORIGIN, z=-2.6)
        dots = VGroup(*[Dot3D(p, radius=0.1, color=K) for p in P])
        qd = sphere(0.18, q)
        ax = arrow3d(np.array([-3.0, 0, -2.6]), np.array([3.2, 0, -2.6]), color=ACCENT)
        self.play(FadeIn(dots), FadeIn(qd), FadeIn(ax), run_time=1.0)
        self.keep(dots, qd, ax)
        tags(self, [(r"$x_1$ (relevant)", np.array([3.6, 0, -2.6])),
                    ("query", q + np.array([0, 0, 0.6]))], 24)
        lines = VGroup(*[Line(q, P[i], color=ACCENT, stroke_width=3) for i in near])
        self.play(*[lit(dots[i]) for i in near], lit(qd), *[Create(l) for l in lines], run_time=1.0)
        self.keep(lines)
        err = float(np.mean(np.abs(P[near, 0] - q[0])))
        tags(self, [("nearest in all 100 dims: wrong $y$", np.array([0, 0, 3.2]))], 24)
        self.orbit(0.5, run=2.0)
        self.wait(0.4)
        self.play(FadeOut(lines), *[unlit(dots[i]) for i in near], unlit(qd), run_time=0.6)
        slab = plane_poly(np.array([0.0, 0, 0]), RIGHT, 5.5, 0.15)
        self.play(FadeIn(slab), *[lit(dots[i]) for i in true], run_time=0.8)
        self.keep(slab)
        tags(self, [(r"neighbors by $x_1$ alone: correct $y$", np.array([0, 0, -3.4]))], 24)
        self.orbit(0.5, run=2.0)
        self.wait(0.6)
        self.clear_scene()
