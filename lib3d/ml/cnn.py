"""CNN components: convolution, pooling, CNN stack."""
from manim import *
import numpy as np
from lib3d.core.engine3d import *
from lib3d.ml.helpers import *
from lib3d.ml.helpers import _u


class ConvSweep:
    def __init__(self, n=6, k=3, cell=0.42, gap_x=4.2, at=ORIGIN):
        at = np.array(at, dtype=float)
        self.n, self.k, self.cell, self.m = n, k, cell, n - k + 1
        self.in_at = at + np.array([-gap_x / 2, 0.0, 0.0])
        self.out_at = at + np.array([gap_x / 2, 0.0, 0.0])
        self.inp = matrix_grid(n, n, cell, self.in_at)
        self.out = matrix_grid(self.m, self.m, cell, self.out_at)
        self.window = Square(side_length=k * cell, color=ACCENT, stroke_width=5)
        self.window.set_fill(ACCENT, opacity=0.12)
        self.window.move_to(self._wc(0, 0))
        self.group = VGroup(self.inp, self.out, self.window)

    def _wc(self, i, j):
        x = (j + (self.k - 1) / 2 - (self.n - 1) / 2) * self.cell
        y = ((self.n - 1) / 2 - (i + (self.k - 1) / 2)) * self.cell
        return self.in_at + np.array([x, y, 0.02])

    def reveal(self, scene, run=1.2):
        scene.play(FadeIn(self.inp), FadeIn(self.out), run_time=run)
        scene.play(FadeIn(self.window), run_time=0.4)
        scene.keep(self.group)
        c = self.cell
        tags(scene, [("input", self.in_at + np.array([0, -(self.n * c / 2 + 0.45), 0])),
                     ("feature map", self.out_at + np.array([0, -(self.m * c / 2 + 0.45), 0]))])

    def sweep(self, scene, step=0.18):
        for i in range(self.m):
            for j in range(self.m):
                cell = grid_cell(self.out, self.m, i, j)
                scene.play(self.window.animate.move_to(self._wc(i, j)), lit(cell), run_time=step)

    def reset(self, scene, run=0.5):
        scene.play(*[unlit(c) for c in self.out], self.window.animate.move_to(self._wc(0, 0)), run_time=run)


class PoolDemo:
    def __init__(self, vals=None, cell=0.55, gap_x=3.8, at=ORIGIN):
        v = np.array(vals if vals is not None else
                     [[1, 3, 2, 1], [4, 6, 5, 2], [3, 1, 0, 7], [2, 8, 4, 3]], dtype=float)
        self.v, self.n, self.cell = v, v.shape[0], cell
        self.m = self.n // 2
        at = np.array(at, dtype=float)
        self.in_at = at + np.array([-gap_x / 2, 0.0, 0.0])
        self.out_at = at + np.array([gap_x / 2, 0.0, 0.0])
        self.inp = matrix_grid(self.n, self.n, cell, self.in_at)
        self.out = matrix_grid(self.m, self.m, cell, self.out_at)
        self.win = Square(side_length=2 * cell, color=ACCENT, stroke_width=5)
        self.win.set_fill(ACCENT, opacity=0.12)
        self.win.move_to(self._block(0, 0))
        self.group = VGroup(self.inp, self.out, self.win)

    def _cp(self, at, rows, cols, i, j, z=0.05):
        return at + np.array([(j - (cols - 1) / 2) * self.cell, ((rows - 1) / 2 - i) * self.cell, z])

    def _block(self, bi, bj):
        n, c = self.n, self.cell
        return self.in_at + np.array([(2 * bj + 0.5 - (n - 1) / 2) * c, ((n - 1) / 2 - (2 * bi + 0.5)) * c, 0.02])

    def reveal(self, scene, run=1.2):
        scene.play(FadeIn(self.inp), FadeIn(self.out), FadeIn(self.win), run_time=run)
        scene.keep(self.group)
        n = self.n
        tags(scene, [(str(int(self.v[i, j])), self._cp(self.in_at, n, n, i, j)) for i in range(n) for j in range(n)], 22)
        tags(scene, [("max pooling", self.out_at + np.array([0, -(self.m * self.cell / 2 + 0.5), 0]))])

    def pool(self, scene, step=0.7):
        for bi in range(self.m):
            for bj in range(self.m):
                scene.play(self.win.animate.move_to(self._block(bi, bj)), run_time=step * 0.5)
                blk = self.v[2 * bi:2 * bi + 2, 2 * bj:2 * bj + 2]
                mi, mj = np.unravel_index(np.argmax(blk), blk.shape)
                mc = grid_cell(self.inp, self.n, 2 * bi + mi, 2 * bj + mj)
                oc = grid_cell(self.out, self.m, bi, bj)
                scene.play(lit(mc), lit(oc), run_time=step * 0.4)
                tags(scene, [(str(int(blk.max())), self._cp(self.out_at, self.m, self.m, bi, bj))], 24, 0.2)


class CNN3D:
    def __init__(self, spec=None, gap=1.7, at=ORIGIN, max_slabs=5):
        spec = spec or (("conv", 1.8, 4), ("pool", 1.2, 4), ("conv", 1.0, 6), ("dense", 8), ("dense", 3))
        at = np.array(at, dtype=float); L = len(spec)
        self.spec, self.layers, self.centers, self.tag_z = spec, [], [], []
        for i, s in enumerate(spec):
            c = at + np.array([(i - (L - 1) / 2) * gap, 0.0, 0.0]); self.centers.append(c)
            if s[0] in ("conv", "pool"):
                size, m = s[1], min(s[2], max_slabs)
                grp = VGroup(*[box((0.1, size, size), c + np.array([(k - (m - 1) / 2) * 0.2, 0, 0]), sw=2)
                               for k in range(m)])
                self.tag_z.append(-(size / 2 + 0.45))
            else:
                h = min(0.14 * s[1], 3.0)
                grp = VGroup(box((0.18, 0.3, h), c, sw=2))
                self.tag_z.append(-(h / 2 + 0.45))
            self.layers.append(grp)
        self.links = [Line(self.centers[i], self.centers[i + 1], color=K, stroke_width=1.5, stroke_opacity=0.5)
                      for i in range(L - 1)]
        self.group = VGroup(*self.layers, *self.links)

    def reveal(self, scene, run=2.0):
        scene.play(LaggedStart(*[FadeIn(g) for g in self.layers], lag_ratio=0.3), run_time=run * 0.6)
        if self.links:
            scene.play(*[Create(l) for l in self.links], run_time=run * 0.4)
        scene.keep(self.group)
        names = {"conv": "conv", "pool": "pool", "dense": "fc"}
        tags(scene, [(names[s[0]], self.centers[i] + np.array([0, 0, self.tag_z[i]]))
                     for i, s in enumerate(self.spec)], 22)

    def forward(self, scene, step=0.6):
        for i, g in enumerate(self.layers):
            extra = []
            if i > 0:
                extra = [ShowPassingFlash(Line(self.centers[i - 1], self.centers[i], color=ACCENT, stroke_width=5),
                                          time_width=0.8)]
            scene.play(*[lit(m) for m in g], *extra, run_time=step)

    def reset(self, scene, run=0.5):
        scene.play(*[unlit(m) for g in self.layers for m in g], run_time=run)
