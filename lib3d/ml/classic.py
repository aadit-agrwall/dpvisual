"""Classic ML: logistic regression, k-means, SVM, trees, embeddings."""
from manim import *
import numpy as np
from lib3d.core.engine3d import *
from lib3d.ml.helpers import *
from lib3d.ml.helpers import _u


def sigmoid_surface(w=(1.2, 0.8), b=0.0, rng=3.0, height=2.0, res=(24, 24)):
    f = lambda u, v: height / (1 + np.exp(-(w[0] * u + w[1] * v + b)))
    return surface(f, (-rng, rng), (-rng, rng), res)


def boundary_line(w=(1.2, 0.8), b=0.0, rng=3.0, z=0.0, dashed=False, width=5):
    n = np.array(w, dtype=float); p0 = -b * n / np.dot(n, n)
    d = np.array([-n[1], n[0]]) / np.linalg.norm(n)
    ts = np.linspace(-12, 12, 4801)
    pts = p0[None, :] + ts[:, None] * d[None, :]
    mask = (np.abs(pts) <= rng).all(axis=1)
    if not mask.any():
        raise ValueError("boundary_line: line is outside the plotted range")
    a, c = pts[mask][0], pts[mask][-1]
    return segment([a[0], a[1], z], [c[0], c[1], z], width=width, dashed=dashed)


def blobs2d(n=12, seed=2, sep=2.4, spread=0.55, direction=(1, 1), z=0.06):
    rng = np.random.default_rng(seed); d = _u(list(direction) + [0.0])[:2]
    pos_pts = [np.array([*(d * sep / 2 + rng.normal(0, spread, 2)), z]) for _ in range(n)]
    neg_pts = [np.array([*(-d * sep / 2 + rng.normal(0, spread, 2)), z]) for _ in range(n)]
    pos = VGroup(*[Dot3D(p, radius=0.11, color=K) for p in pos_pts])
    neg = VGroup(*[box((0.22, 0.22, 0.22), p, sw=1.5) for p in neg_pts])
    return pos, neg, pos_pts, neg_pts


class KMeans3D:
    def __init__(self, k=3, per=8, seed=4, spread=0.45, at=ORIGIN):
        at = np.array(at, dtype=float); rng = np.random.default_rng(seed)
        preset = [(-2, -1, 0.5), (2, -1, 0.0), (0, 2, 1.0), (0, -2, 1.5)]
        centers = np.array(preset[:k], dtype=float)
        pts = np.vstack([c + rng.normal(0, spread, (per, 3)) for c in centers]) + at
        self.k, self.pts = k, pts
        cent = pts[[0, 1, per, per + 1][:k]].copy()
        self.init = [c.copy() for c in cent]; self.steps = []
        for _ in range(6):
            d = np.linalg.norm(pts[:, None, :] - cent[None, :, :], axis=2); a = d.argmin(1)
            new = np.array([pts[a == j].mean(0) if (a == j).any() else cent[j] for j in range(k)])
            self.steps.append((a.copy(), new.copy()))
            if np.allclose(new, cent):
                break
            cent = new
        self.dots = VGroup(*[Dot3D(p, radius=0.1, color=K) for p in pts])
        self.cents = [box((0.3, 0.3, 0.3), c) for c in self.init]
        self.cur = [c.copy() for c in self.init]
        self.group = VGroup(self.dots, *self.cents)

    def reveal(self, scene, run=1.2):
        scene.play(FadeIn(self.dots), run_time=run)
        scene.play(*[GrowFromCenter(c) for c in self.cents], run_time=0.6)
        scene.keep(self.group)

    def step(self, scene, i, run=1.0):
        a, new = self.steps[i]
        lines = VGroup(*[Line(self.pts[n], self.cur[a[n]], color=K, stroke_width=1.5, stroke_opacity=0.6)
                         for n in range(len(self.pts))])
        scene.play(Create(lines, lag_ratio=0.02), run_time=run * 0.6)
        scene.play(*[self.cents[j].animate.move_to(new[j]).set_fill(ACCENT, opacity=0.9) for j in range(self.k)],
                   run_time=run * 0.8)
        scene.play(FadeOut(lines), *[unlit(c) for c in self.cents], run_time=run * 0.4)
        self.cur = [new[j].copy() for j in range(self.k)]


class SVMMargin3D:
    def __init__(self, n=8, seed=3, sep=3.0, spread=0.4, size=2.8, at=ORIGIN):
        at = np.array(at, dtype=float); u = _u((1, 1, 1)); self.u = u
        self.pos, self.neg, pp, nn = scatter_classes(n, seed, sep, spread, at)
        dp = np.array([float(np.dot(p - at, u)) for p in pp])
        dn = np.array([float(np.dot(p - at, u)) for p in nn])
        self.ip = int(np.argmin(np.abs(dp))); self.in_ = int(np.argmin(np.abs(dn)))
        m_pos, m_neg = abs(dp[self.ip]), abs(dn[self.in_])
        off = (m_pos - m_neg) / 2
        self.plane = plane_poly(at + u * off, u, size, 0.22)
        half = max((m_pos + m_neg) / 2, 0.1)
        self.mp = [plane_poly(at + u * (off + half), u, size, 0.08), plane_poly(at + u * (off - half), u, size, 0.08)]
        self.group = VGroup(self.pos, self.neg, self.plane)

    def reveal(self, scene, run=1.4):
        scene.play(FadeIn(self.pos), FadeIn(self.neg), FadeIn(self.plane), run_time=run)
        scene.keep(self.group)

    def show_margin(self, scene, run=1.0):
        scene.play(*[FadeIn(p) for p in self.mp], run_time=run)
        scene.keep(*self.mp)

    def highlight_support(self, scene, run=0.8):
        scene.play(lit(self.pos[self.ip]), lit(self.neg[self.in_]), run_time=run)


class DecisionTree3D:
    def __init__(self, depth=3, width=6.0, drop=1.4, at=ORIGIN):
        at = np.array(at, dtype=float); self.depth = depth; self.pos = {}
        for d in range(depth + 1):
            for i in range(2 ** d):
                self.pos[(d, i)] = at + np.array([(i + 0.5) * width / 2 ** d - width / 2, 0, 1.6 - d * drop])
        self.mob = {k: (sphere(0.22, p) if k[0] < depth else box((0.3, 0.3, 0.3), p)) for k, p in self.pos.items()}
        self.lines = {}
        for (d, i), p in self.pos.items():
            if d < depth:
                for c in (0, 1):
                    ch = (d + 1, 2 * i + c)
                    self.lines[((d, i), ch)] = Line(p, self.pos[ch], color=K, stroke_width=3)
        self.group = VGroup(*self.mob.values(), *self.lines.values())

    def reveal(self, scene, run=1.6):
        scene.play(*[FadeIn(m) for m in self.mob.values()], *[Create(l) for l in self.lines.values()], run_time=run)
        scene.keep(self.group)
        tags(scene, [("$x_1<t$", self.pos[(0, 0)] + np.array([0, 0, 0.5]))], 24)

    def fire_path(self, scene, path=(0, 1, 0), step=0.6):
        if len(path) > self.depth:
            raise ValueError("path longer than tree depth")
        key = (0, 0)
        scene.play(lit(self.mob[key]), run_time=step * 0.4)
        for c in path:
            child = (key[0] + 1, 2 * key[1] + c)
            flashes(scene, [(self.pos[key], self.pos[child])], run=step)
            scene.play(lit(self.mob[child]), run_time=step * 0.4)
            key = child

    def reset(self, scene, run=0.5):
        scene.play(*[unlit(m) for m in self.mob.values()], run_time=run)


def embedding_cloud(k=3, per=9, seed=5, spread=0.5, rng_=2.2, at=ORIGIN):
    at = np.array(at, dtype=float); rng = np.random.default_rng(seed)
    centers = rng.uniform(-rng_, rng_, (k, 3))
    pts = [at + centers[j] + rng.normal(0, spread, 3) for j in range(k) for _ in range(per)]
    return VGroup(*[Dot3D(p, radius=0.09, color=K) for p in pts]), pts


def show_neighbors(scene, grp, pts, idx=0, k=3, run=1.0):
    d = [float(np.linalg.norm(p - pts[idx])) for p in pts]
    order = [int(j) for j in np.argsort(d)[1:k + 1]]
    lines = VGroup(*[Line(pts[idx], pts[j], color=ACCENT, stroke_width=3) for j in order])
    scene.play(lit(grp[idx]), *[lit(grp[j]) for j in order], *[Create(l) for l in lines], run_time=run)
    scene.keep(lines)
    return lines


def batch_stack(n=4, size=4, cell=0.32, gap_z=0.5, at=ORIGIN):
    return VGroup(*[matrix_grid(size, size, cell, at, z=(k - (n - 1) / 2) * gap_z) for k in range(n)])
