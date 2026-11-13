from manim import *
from lib3d import *
import numpy as np

BLUE_ = "#1F4FFF"
INK = "#000000"


def rotz(v, a):
    c, s = np.cos(a), np.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1], v[2]])


def dots(points, color=INK, r=0.06):
    return VGroup(*[Dot3D(p, radius=r, color=color) for p in points])


class Page147(Stage3D):
    def construct(self):
        # ---------- 1. PCA: rotation that decorrelates ----------
        self.setup_stage("oblique", zoom=0.85)
        self.caption("PCA: rotate the data so z has diagonal covariance")

        rng = np.random.default_rng(147)
        tilt = np.deg2rad(35)
        raw = rng.normal(size=(60, 3)) * np.array([2.3, 0.9, 0.6])
        pts = np.array([rotz(p, tilt) for p in raw])
        cloud = dots(pts)

        # new basis (fixed, black) and principal axes (move with the data, blue)
        basis = VGroup(
            Line3D(-3.6 * RIGHT, 3.6 * RIGHT, color=INK, thickness=0.01),
            Line3D(-3.6 * UP, 3.6 * UP, color=INK, thickness=0.01),
            Line3D(-2.4 * OUT, 2.4 * OUT, color=INK, thickness=0.01),
        )
        axes_pc = VGroup(
            Line3D(rotz(-3.4 * RIGHT, tilt), rotz(3.4 * RIGHT, tilt), color=BLUE_, thickness=0.02),
            Line3D(rotz(-2.2 * UP, tilt), rotz(2.2 * UP, tilt), color=BLUE_, thickness=0.02),
        )
        moving = VGroup(cloud, axes_pc)

        self.reveal(basis, anim="create", run=1.2)
        self.reveal(moving, anim="fade", run=1.5)
        self.keep(basis, moving)
        self.orbit(PI / 6, run=2.5)
        self.play(Rotate(moving, angle=-tilt, axis=OUT, about_point=ORIGIN),
                  run_time=3.5, rate_func=smooth)
        self.label3d("Cov(z) = Σ²  (diagonal)", np.array([0, -3.4, 2.6]))
        self.orbit(PI / 6, run=3)
        self.wait(1)
        self.clear_scene()

        # ---------- 2. k-means: one-hot code h ----------
        self.caption("k-means: x in cluster i gives h_i = 1, all other h = 0")
        centers = [np.array([-3.2, -1.0, 0.4]),
                   np.array([2.8, -2.0, -0.4]),
                   np.array([0.4, 2.8, 1.0])]
        groups = []
        for c in centers:
            groups.append(dots(c + rng.normal(size=(14, 3)) * 0.55))
        floor = VGroup(*[Line3D(np.array([-5, y, -1.6]), np.array([5, y, -1.6]),
                                color=INK, thickness=0.005) for y in range(-5, 6, 2)],
                       *[Line3D(np.array([x, -5, -1.6]), np.array([x, 5, -1.6]),
                                color=INK, thickness=0.005) for x in range(-5, 6, 2)])
        self.reveal(floor, anim="create", run=1.0)
        self.reveal(*groups, anim="fade", run=1.5)
        self.keep(floor, *groups)
        self.orbit(PI / 6, run=2.5)

        codes = ["h = (1, 0, 0)", "h = (0, 1, 0)", "h = (0, 0, 1)"]
        for g, c, t in zip(groups, centers, codes):
            ball = Sphere(radius=0.2, resolution=(12, 12), fill_color=BLUE_, fill_opacity=1,
                          stroke_width=0).move_to(c)
            self.play(g.animate.set_color(BLUE_), FadeIn(ball), run_time=1.0)
            self.keep(ball)
            self.label3d(t, c + np.array([0, 0, 1.6]))
            self.wait(0.8)
            self.play(g.animate.set_color(INK), run_time=0.5)
        self.orbit(PI / 6, run=2.5)
        self.clear_scene()

        # ---------- 3. one-hot = sparse representation ----------
        self.caption("A one-hot code is sparse: most entries are zero")
        k = 8
        hot = 2
        cubes = VGroup(*[
            Cube(side_length=0.6, fill_color=WHITE, fill_opacity=1,
                 stroke_color=INK, stroke_width=1.5).move_to(np.array([(i - (k - 1) / 2) * 0.9, 0, 0]))
            for i in range(k)])
        self.reveal(cubes, anim="fade", run=1.2)
        self.keep(cubes)
        for i in range(k):
            self.label3d("1" if i == hot else "0", np.array([(i - (k - 1) / 2) * 0.9, 0, 0.8]))
        self.play(cubes[hot].animate.set_fill(BLUE_, opacity=1), run_time=1.0)
        self.orbit(PI / 5, run=3)
        self.wait(1)
        self.clear_scene()
