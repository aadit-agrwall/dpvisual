"""Chapter 5 estimation: bias/variance, MLE, Bayesian update, kernel trick, smoothness.
Every reveal() sets its own 3D camera and adds depth. No edge-on layouts."""
from manim import *
import numpy as np
from lib3d.core.engine3d import *
from lib3d.ml.helpers import *
from lib3d.ml.basics_fit import _stage_floor


def _swap_tag(scene, old, text, pos, size=28):
    if old:
        scene.play(*[FadeOut(m) for m in old], run_time=0.3)
    return tags(scene, [(text, pos)], size)


class BiasVariance3D:
    """m datasets, each on its own depth layer. fit(degree) shows the spread of fits."""
    def __init__(self, m=6, n=10, seed=4, noise=0.35, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        self.f = lambda x: 0.9 * np.sin(0.9 * x)
        self.xs = np.linspace(-2.6, 2.6, n)
        self.data = [self.f(self.xs) + rng.normal(0, noise, n) for _ in range(m)]
        self.depth = [(k - (m - 1) / 2) * 0.7 for k in range(m)]
        a = self.at
        self.truth = VGroup(*[curve3d(lambda t, e=e: a + np.array([t, e, float(self.f(t))]), (-2.8, 2.8), K, 2)
                              for e in self.depth])
        self.dots = VGroup(*[Dot3D(a + np.array([x, e, y]), radius=0.07, color=K)
                             for ys, e in zip(self.data, self.depth) for x, y in zip(self.xs, ys)])
        self.group = VGroup(self.truth, self.dots)
        self.sheet, self.tag, self.P = None, [], []

    def reveal(self, scene, run=1.2):
        scene.view("oblique", 1.0)
        _stage_floor(scene, self.at)
        scene.play(Create(self.truth), FadeIn(self.dots), run_time=run)
        scene.keep(self.group)

    def fit(self, scene, degree, run=1.0):
        a = self.at
        self.P = [np.poly1d(np.polyfit(self.xs, ys, degree)) for ys in self.data]
        sheet = VGroup(*[curve3d(lambda t, p=p, e=e: a + np.array([t, e, float(np.clip(p(t), -2.6, 2.6))]),
                                 (-2.8, 2.8), ACCENT, 3) for p, e in zip(self.P, self.depth)])
        old = ([self.sheet] if self.sheet is not None else []) + self.tag
        if old:
            scene.play(*[FadeOut(m) for m in old], run_time=0.4)
        scene.play(*[Create(c) for c in sheet], run_time=run)
        scene.keep(sheet)
        scene.orbit(0.35, run=1.2)
        self.sheet = sheet
        label = "high bias, low variance" if degree <= 1 else ("high variance" if degree >= 7 else "balanced")
        self.tag = tags(scene, [(label, a + np.array([0, 0, -3.0]))], 28)

    def show_mean(self, scene, run=1.0):
        a, e = self.at, -2.5
        pm = lambda t: float(np.clip(np.mean([p(t) for p in self.P]), -2.6, 2.6))
        tr = curve3d(lambda t: a + np.array([t, e, float(self.f(t))]), (-2.8, 2.8), K, 6)
        mn = curve3d(lambda t: a + np.array([t, e, pm(t)]), (-2.8, 2.8), ACCENT, 6)
        scene.play(Create(tr), Create(mn), run_time=run)
        scene.keep(tr, mn)
        tags(scene, [("mean fit vs truth", a + np.array([0, e, 2.4]))], 24)


class MLE3D:
    """Gaussian likelihood surface over (mu, sigma). ascend() climbs to the maximum."""
    def __init__(self, n=14, seed=6, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        self.x = rng.normal(0.5, 1.0, n)
        self.mu0, self.s0 = float(self.x.mean()), float(self.x.std())

        def ll(mu, s):
            acc = 0.0
            for xi in self.x:
                acc = acc - np.log(s) - (xi - mu) ** 2 / (2 * s * s)
            return acc
        llmax = ll(self.mu0, self.s0)
        self.z = lambda u, v: 3.0 * np.exp((ll(u, 1.4 + v) - llmax) / (n * 0.35))
        self.surf = surface(self.z, (-2.5, 2.5), (-1.0, 1.0), (26, 26)).shift(self.at)
        self.data = VGroup(*[Dot3D(self.at + np.array([xi, -1.35, 0.05]), radius=0.08, color=K) for xi in self.x])
        self.group = VGroup(self.surf, self.data)

    def reveal(self, scene, run=1.6):
        scene.view("oblique", 1.0)
        _stage_floor(scene, self.at, z=-0.05)
        scene.play(Create(self.surf), FadeIn(self.data), run_time=run)
        scene.keep(self.group)
        a = self.at
        tags(scene, [(r"$\mu$", a + np.array([2.9, -1.5, 0])), (r"$\sigma$", a + np.array([-2.8, 0, 0])),
                     ("data", a + np.array([0, -1.9, 0.0]))], 24)

    def ascend(self, scene, start=(-1.8, 0.8), steps=14, step_run=0.22, eps=1e-4):
        x = np.array(start, dtype=float)
        pts = [x.copy()]
        for k in range(steps):
            g = np.array([(self.z(x[0] + eps, x[1]) - self.z(x[0] - eps, x[1])) / (2 * eps),
                          (self.z(x[0], x[1] + eps) - self.z(x[0], x[1] - eps)) / (2 * eps)])
            n = np.linalg.norm(g)
            x = np.clip(x + (g / n if n > 1e-9 else 0) * 0.45 * (0.88 ** k), [-2.5, -1.0], [2.5, 1.0])
            pts.append(x.copy())
        P = [self.at + np.array([p[0], p[1], float(self.z(p[0], p[1])) + 0.08]) for p in pts]
        dot = Dot3D(P[0], radius=0.14, color=ACCENT)
        scene.play(FadeIn(dot), run_time=0.4)
        trail = VGroup()
        for p, q in zip(P[:-1], P[1:]):
            seg = segment(p, q, width=4, color=ACCENT)
            scene.play(dot.animate.move_to(q), Create(seg), run_time=step_run)
            trail.add(seg)
        scene.keep(dot, trail)
        tags(scene, [("maximum likelihood", P[-1] + np.array([0, 0, 0.7]))], 24)
        return P


class Bayesian3D:
    """Prior -> posterior. Each data update is one curve on a new depth layer."""
    def __init__(self, counts=(0, 1, 4, 16), prior=(0.0, 1.5), sigma=1.0, truth=1.0, seed=2, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        data = rng.normal(truth, sigma, max(counts))
        self.truth, self.counts = truth, counts
        L = len(counts)
        self.depth = [(k - (L - 1) / 2) * 1.0 for k in range(L)]
        a, self.layers, self.names = self.at, [], []
        for n, e in zip(counts, self.depth):
            prec = 1 / prior[1] ** 2 + n / sigma ** 2
            mean = (prior[0] / prior[1] ** 2 + data[:n].sum() / sigma ** 2) / prec
            sd = prec ** -0.5
            pdf = lambda t, m=mean, s=sd: 1.6 * np.exp(-0.5 * ((t - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
            self.layers.append((lambda c, e=e, pdf=pdf: curve3d(
                lambda t: a + np.array([t, e, float(min(pdf(t), 3.2))]), (-3, 3), c, 5)))
            self.names.append("prior" if n == 0 else f"{n} samples")
        self.i, self.prev = 0, None

    def reveal(self, scene, run=1.0):
        scene.view("oblique", 1.0)
        _stage_floor(scene, self.at, z=-0.05)
        a, v = self.at, self.truth
        ln = segment(a + np.array([v, -2.0, 0]), a + np.array([v, 2.0, 0]), width=3, dashed=True)
        scene.play(Create(ln), run_time=run)
        scene.keep(ln)
        tags(scene, [("true value", a + np.array([v, -2.4, 0.0]))], 24)

    def update(self, scene, run=0.9):
        if self.i >= len(self.layers):
            return
        cur = self.layers[self.i](ACCENT)
        anims = [Create(cur)]
        if self.prev is not None:
            anims.append(self.prev.animate.set_color(K))
        scene.play(*anims, run_time=run)
        scene.keep(cur)
        tags(scene, [(self.names[self.i], self.at + np.array([3.4, self.depth[self.i], 0.2]))], 24)
        self.prev = cur
        self.i += 1


class KernelTrick3D:
    """Rings are not linearly separable in 2D. lift() adds z = r^2, then a plane separates them."""
    def __init__(self, n=16, seed=2, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        self.k, self.zt = 0.35, 0.35 * 1.5 ** 2
        ang = rng.uniform(0, 2 * np.pi, (2, n))
        r_in, r_out = rng.uniform(0.2, 1.0, n), rng.uniform(2.0, 2.7, n)
        self.xy = [(r * np.cos(t), r * np.sin(t)) for r, t in zip(r_in, ang[0])] + \
                  [(r * np.cos(t), r * np.sin(t)) for r, t in zip(r_out, ang[1])]
        self.mobs = [box((0.22, 0.22, 0.22), self.at + np.array([x, y, 0.06]), sw=1.5) for x, y in self.xy[:n]] + \
                    [Dot3D(self.at + np.array([x, y, 0.06]), radius=0.11, color=K) for x, y in self.xy[n:]]
        self.group = VGroup(*self.mobs)
        self.plane = plane_poly(self.at + np.array([0, 0, self.zt]), OUT, 6.5, 0.2)

    def reveal(self, scene, run=1.2):
        scene.view("oblique", 1.0)
        _stage_floor(scene, self.at, z=0.0)
        scene.play(FadeIn(self.group), run_time=run)
        scene.keep(self.group)

    def lift(self, scene, run=1.6):
        a = self.at
        scene.play(*[m.animate.move_to(a + np.array([x, y, self.k * (x * x + y * y)]))
                     for m, (x, y) in zip(self.mobs, self.xy)], run_time=run)
        scene.play(FadeIn(self.plane), run_time=0.6)
        scene.keep(self.plane)
        scene.orbit(0.5, run=1.6)
        tags(scene, [(r"$z=x^2+y^2$", a + np.array([0, 0, 3.4]))], 26)

    def unlift(self, scene, run=1.4):
        a = self.at
        circ = curve3d(lambda t: a + np.array([1.5 * np.cos(t), 1.5 * np.sin(t), 0.03]), (0, 2 * np.pi), ACCENT, 5)
        scene.play(FadeOut(self.plane), *[m.animate.move_to(a + np.array([x, y, 0.06]))
                                          for m, (x, y) in zip(self.mobs, self.xy)], run_time=run)
        scene.play(Create(circ), run_time=0.8)
        scene.keep(circ)


class Smoothness3D:
    """Bumpy surface through noisy points (overfit) vs smooth surface (regularized)."""
    def __init__(self, seed=3, at=ORIGIN):
        self.at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        sm = lambda u, v: 0.9 + 0.5 * np.sin(0.6 * u) * np.cos(0.6 * v)
        bm = lambda u, v: sm(u, v) + 0.35 * np.sin(3 * u) * np.sin(3 * v)
        self.smooth = surface(sm, (-2.5, 2.5), (-2.5, 2.5), (22, 22)).shift(self.at)
        self.bumpy = surface(bm, (-2.5, 2.5), (-2.5, 2.5), (32, 32)).shift(self.at)
        uv = rng.uniform(-2.2, 2.2, (9, 2))
        self.dots = VGroup(*[Dot3D(self.at + np.array([u, v, float(bm(u, v))]), radius=0.1, color=ACCENT)
                             for u, v in uv])
        self.tag = []

    def reveal(self, scene, run=1.2):
        scene.view("oblique", 1.0)
        _stage_floor(scene, self.at, z=-0.1)
        scene.play(FadeIn(self.dots), run_time=run)
        scene.keep(self.dots)

    def show_bumpy(self, scene, run=1.4):
        scene.play(Create(self.bumpy), run_time=run)
        scene.keep(self.bumpy)
        self.tag = _swap_tag(scene, self.tag, "overfit: bumpy, passes through every point",
                             self.at + np.array([0, 0, 2.8]), 26)
        scene.orbit(0.4, run=1.4)

    def regularize(self, scene, run=1.4):
        scene.play(FadeOut(self.bumpy), FadeIn(self.smooth), run_time=run)
        scene.keep(self.smooth)
        self.tag = _swap_tag(scene, self.tag, "smooth: nearby inputs, nearby outputs",
                             self.at + np.array([0, 0, 2.8]), 26)
        scene.orbit(0.4, run=1.4)
