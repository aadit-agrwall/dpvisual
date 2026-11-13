from manim import *
import numpy as np
from lib3d import *
from lib3d.ml.basics_fit import _stage_floor


class Page142(Stage3D):
    def construct(self):
        self.setup_stage("oblique")

        # 1. nearest neighbor is decided by x2..x100, not by the lone feature x1
        self.caption(r"Nearest neighbor ignores that only $x_1$ matters")
        self.view("oblique", 1.0)
        rng = np.random.default_rng(5)
        P = rng.uniform(-2.4, 2.4, (26, 3))
        q = np.zeros(3)
        d = np.linalg.norm(P[:, 1:] - q[1:], axis=1)        # distance WITHOUT x1
        near = [int(i) for i in np.argsort(d)[:3]]
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
        tags(self, [(r"nearest by $x_2 \ldots x_{100}$", np.array([0, 0, 3.2]))], 24)
        self.orbit(0.5, run=2.0)
        tags(self, [("small training set: output is random", np.array([0, 0, -3.4]))], 24)
        self.wait(0.5)
        self.clear_scene()

        # 2. decision tree: regions, axis-aligned cuts, leaf = region
        self.caption("Decision tree: each leaf owns one region")
        self.view("slant", 1.0)
        t = DecisionTree3D(width=4.5, at=[-2.6, 0, 0.3]); t.reveal(self)
        C = np.array([3.2, 0.0, 0.0]); h = 1.3
        corners = [C + np.array([-h, 0, -h]), C + np.array([h, 0, -h]), C + np.array([h, 0, h]), C + np.array([-h, 0, h])]
        frame = [segment(corners[i], corners[(i + 1) % 4], width=4) for i in range(4)]
        cuts = [segment(C + np.array([-0.3, 0, -h]), C + np.array([-0.3, 0, h]), width=3),
                segment(C + np.array([-0.3, 0, 0.2]), C + np.array([h, 0, 0.2]), width=3),
                segment(C + np.array([-h, 0, -0.5]), C + np.array([-0.3, 0, -0.5]), width=3)]
        self.play(*[Create(m) for m in frame + cuts], run_time=1.2)
        self.keep(*frame, *cuts)
        tags(self, [("axis-aligned cuts", C + np.array([0, 0, 1.9])),
                    ("leaf = region", C + np.array([0, 0, -1.9]))], 24)
        t.fire_path(self, (0, 1, 0), 0.5)
        self.orbit(0.5, run=2.0)
        self.clear_scene()

        # 3. trees struggle when the boundary is x2 > x1 (not axis aligned)
        self.caption(r"Positive class where $x_2 > x_1$")
        self.view("oblique", 1.0)
        _stage_floor(self, ORIGIN, z=-0.05)
        r2 = np.random.default_rng(8)
        Q = r2.uniform(-2.4, 2.4, (30, 2))
        mobs = []
        for x, y in Q:
            if y > x + 0.25:
                mobs.append(Dot3D(np.array([x, y, 0.08]), radius=0.11, color=K))
            elif y < x - 0.25:
                mobs.append(box((0.22, 0.22, 0.22), np.array([x, y, 0.08]), sw=1.5))
        a1 = arrow3d(np.array([-3.0, -3.0, 0.0]), np.array([3.2, -3.0, 0.0]), color=K)
        a2 = arrow3d(np.array([-3.0, -3.0, 0.0]), np.array([-3.0, 3.2, 0.0]), color=K)
        self.play(*[FadeIn(m) for m in mobs + [a1, a2]], run_time=1.0)
        self.keep(*mobs, a1, a2)
        tags(self, [(r"$x_1$", np.array([3.6, -3.0, 0])), (r"$x_2$", np.array([-3.0, 3.6, 0]))], 24)
        cut = segment(np.array([0, -2.6, 0.06]), np.array([0, 2.6, 0.06]), width=5, color=ACCENT)
        self.play(Create(cut), run_time=0.8)
        self.keep(cut)
        tags(self, [("one axis-aligned cut: wrong boundary", np.array([0, 0, 2.6]))], 24)
        self.orbit(0.4, run=1.6)
        self.play(FadeOut(cut), run_time=0.4)
        s = 0.65
        cur = (-2.6, -2.6); pts = [np.array([cur[0], cur[1], 0.06])]
        for _ in range(8):
            cur = (cur[0] + s, cur[1]); pts.append(np.array([cur[0], cur[1], 0.06]))
            cur = (cur[0], cur[1] + s); pts.append(np.array([cur[0], cur[1], 0.06]))
        stairs = VMobject(color=ACCENT, stroke_width=5).set_points_as_corners(pts)
        self.play(Create(stairs), run_time=2.0)
        self.keep(stairs)
        tags(self, [("tree: many nodes, a staircase of steps", np.array([0, 0, 3.0]))], 24)
        self.orbit(0.4, run=1.6)
        self.play(FadeOut(stairs), run_time=0.4)
        diag = segment(np.array([-2.6, -2.6, 0.06]), np.array([2.6, 2.6, 0.06]), width=5, color=ACCENT)
        self.play(Create(diag), run_time=1.0)
        self.keep(diag)
        tags(self, [("logistic regression: one line, correct boundary", np.array([0, 0, 3.0]))], 24)
        self.orbit(0.4, run=1.6)
        self.wait(0.4)
        self.clear_scene()

        # 4. limitations, but useful baselines when compute is limited
        self.caption("Limited, but useful when compute is constrained")
        self.view("slant", 1.0)
        bk = box((0.9, 0.9, 0.9), np.array([-3.0, -1.0, 1.0]))
        bt = box((0.9, 0.9, 0.9), np.array([-3.0, 1.0, -1.0]))
        bs = box((1.3, 1.3, 1.3), np.array([2.5, 0.0, 0.0]))
        self.play(FadeIn(bk), FadeIn(bt), FadeIn(bs), run_time=1.0)
        self.keep(bk, bt, bs)
        tags(self, [("k-NN", np.array([-3.0, -1.0, 2.0])), ("decision tree", np.array([-3.0, 1.0, -2.1])),
                    ("sophisticated algorithm", np.array([2.5, 0.0, 1.5]))], 24)
        flashes(self, [(np.array([-2.5, -1.0, 1.0]), np.array([1.85, 0, 0.1])),
                       (np.array([-2.5, 1.0, -1.0]), np.array([1.85, 0, -0.1]))], run=0.8)
        self.play(lit(bk), lit(bt), lit(bs), run_time=0.6)
        tags(self, [("compare against these baselines", np.array([0, 0, -2.8]))], 24)
        self.orbit(0.5, run=2.0)
        self.clear_scene()

        # 5. 5.8 unsupervised learning: features only, no supervision signal
        self.caption("5.8  Unsupervised learning: features only")
        self.view("oblique", 1.0)
        _stage_floor(self, ORIGIN, z=-2.0)
        km = KMeans3D(); km.reveal(self)
        tags(self, [("no labels, no supervision signal", np.array([0, 0, 3.0]))], 24)
        km.step(self, 0, 1.0)
        self.orbit(0.5, run=2.0)
        self.clear_scene()
