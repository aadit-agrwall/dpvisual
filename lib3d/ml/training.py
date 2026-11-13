"""Training: computation graph, loss landscape, autoencoder, dropout, backprop, residual block."""
from manim import *
import numpy as np
from lib3d.core.engine3d import *
from lib3d.ml.helpers import *
from lib3d.ml.helpers import _u


class CompGraph3D:
    def __init__(self, nodes, edges, at=ORIGIN):
        at = np.array(at, dtype=float)
        self.pos, self.mob, self.label, self.edges = {}, {}, {}, list(edges)
        for name, (p, kind, label) in nodes.items():
            P = at + np.array(p, dtype=float); self.pos[name] = P; self.label[name] = label
            if kind == "var":
                self.mob[name] = box((0.4, 0.4, 0.4), P)
            elif kind == "op":
                self.mob[name] = sphere(0.28, P)
            else:
                self.mob[name] = box((0.55, 0.55, 0.55), P)
        self.lines = {e: Line(self.pos[e[0]], self.pos[e[1]], color=K, stroke_width=3) for e in self.edges}
        self.group = VGroup(*self.mob.values(), *self.lines.values())
        self.order = self._topo()

    @staticmethod
    def linear_loss(at=ORIGIN):
        nodes = {"x": ([-3, 0, 1.2], "var", "$x$"), "w": ([-3, 0, -1.2], "var", "$w$"),
                 "m": ([-1, 0, 0], "op", r"$\times$"), "b": ([-1, 0, -1.8], "var", "$b$"),
                 "a": ([1, 0, -0.2], "op", "$+$"), "y": ([1, 0, 1.8], "var", "$y$"),
                 "L": ([3, 0, 0.8], "loss", "$L$")}
        edges = [("x", "m"), ("w", "m"), ("m", "a"), ("b", "a"), ("a", "L"), ("y", "L")]
        return CompGraph3D(nodes, edges, at)

    def _topo(self):
        indeg = {n: 0 for n in self.pos}
        for a, b in self.edges:
            indeg[b] += 1
        queue = [n for n, d in indeg.items() if d == 0]; out = []
        while queue:
            n = queue.pop(0); out.append(n)
            for a, b in self.edges:
                if a == n:
                    indeg[b] -= 1
                    if indeg[b] == 0:
                        queue.append(b)
        return out

    def reveal(self, scene, run=1.6):
        scene.play(*[FadeIn(m) for m in self.mob.values()], *[Create(l) for l in self.lines.values()], run_time=run)
        scene.keep(self.group)
        tags(scene, [(self.label[n], self.pos[n] + np.array([0, 0, 0.55])) for n in self.pos], 26)

    def forward(self, scene, step=0.5):
        for n in self.order:
            incoming = [(self.pos[a], self.pos[b]) for a, b in self.edges if b == n]
            flashes(scene, incoming, run=step)
            scene.play(lit(self.mob[n]), run_time=step * 0.4)

    def backward(self, scene, step=0.5):
        for n in reversed(self.order):
            parents = [(self.pos[b], self.pos[a]) for a, b in self.edges if b == n]
            if parents:
                flashes(scene, parents, run=step)
                scene.play(*[lit(self.mob[a]) for a, b in self.edges if b == n], run_time=step * 0.4)

    def reset(self, scene, run=0.5):
        scene.play(*[unlit(m) for m in self.mob.values()], run_time=run)


def descent_path(fn, start, lr=0.1, steps=30, momentum=0.0, eps=1e-4):
    x = np.array(start, dtype=float); v = np.zeros(2); pts = [x.copy()]
    for _ in range(steps):
        g = np.array([(fn(x[0] + eps, x[1]) - fn(x[0] - eps, x[1])) / (2 * eps),
                      (fn(x[0], x[1] + eps) - fn(x[0], x[1] - eps)) / (2 * eps)])
        v = momentum * v - lr * g
        x = x + v; pts.append(x.copy())
    return pts


class LossLandscape:
    def __init__(self, fn=None, rng=2.5, res=(22, 22)):
        self.fn = fn or (lambda u, v: 0.1 * u * u + 0.3 * v * v)
        self.axes = axes3d((-rng, rng, 1), (-rng, rng, 1), (0, 3, 1), (2 * rng, 2 * rng, 3))
        self.surf = surface(self.fn, (-rng, rng), (-rng, rng), res)
        self.group = VGroup(self.axes, self.surf)

    def reveal(self, scene, run=1.6):
        scene.play(Create(self.axes), Create(self.surf), run_time=run)
        scene.keep(self.group)

    def descend(self, scene, start=(2.2, 1.8), lr=2.4, steps=10, momentum=0.0, step_run=0.3, dashed=False):
        pts = descent_path(self.fn, start, lr, steps, momentum)
        P = [np.array([p[0], p[1], self.fn(p[0], p[1]) + 0.07]) for p in pts]
        dot = Dot3D(P[0], radius=0.14, color=ACCENT)
        scene.play(FadeIn(dot), run_time=0.4)
        trail = VGroup()
        for a, b in zip(P[:-1], P[1:]):
            seg = segment(a, b, width=4, dashed=dashed)
            scene.play(dot.animate.move_to(b), Create(seg), run_time=step_run)
            trail.add(seg)
        scene.keep(dot, trail)
        return dot, trail, P


def gradient_arrow(fn, p, scale=1.5, eps=1e-4, descent=True):
    g = np.array([(fn(p[0] + eps, p[1]) - fn(p[0] - eps, p[1])) / (2 * eps),
                  (fn(p[0], p[1] + eps) - fn(p[0], p[1] - eps)) / (2 * eps)])
    n = np.linalg.norm(g)
    d = (g / n if n > 1e-9 else np.zeros(2)) * (-1 if descent else 1) * scale
    a = np.array([p[0], p[1], fn(p[0], p[1]) + 0.07])
    return arrow3d(a, a + np.array([d[0], d[1], 0.0]), color=ACCENT)


class Autoencoder3D(Net3D):
    def __init__(self, enc=(6, 4), code=2, **kw):
        super().__init__(tuple(enc) + (code,) + tuple(reversed(enc)), **kw)
        self.code_idx = len(enc)

    def reveal(self, scene, run=3.0):
        Net3D.reveal(self, scene, run)
        scene.keep(self.group)
        zt = max(p[2] for pts in self.pos for p in pts) + 0.9
        y0 = self.pos[self.code_idx][0][1]
        ex = np.mean([self.pos[i][0][0] for i in range(self.code_idx)])
        dx = np.mean([self.pos[i][0][0] for i in range(self.code_idx + 1, len(self.layers))])
        cx = self.pos[self.code_idx][0][0]
        tags(scene, [("encoder", [ex, y0, zt]), ("code", [cx, y0, zt]), ("decoder", [dx, y0, zt])], 26)

    def squeeze(self, scene, run=0.8):
        scene.play(*[lit(d) for d in self.nodes[self.code_idx]], run_time=run)


def dropout(scene, net, layer, idx, run=0.6):
    scene.play(*[net.nodes[layer][k].animate.set_opacity(0.12) for k in idx], run_time=run)


def undrop(scene, net, layer, idx, run=0.5):
    scene.play(*[net.nodes[layer][k].animate.set_opacity(1.0) for k in idx], run_time=run)


def backprop(scene, net, step=0.6):
    L = len(net.layers)
    scene.play(*[lit(d) for d in net.nodes[L - 1]], run_time=step * 0.6)
    for i in range(L - 1, 0, -1):
        pairs = [(q, p) for p in net.pos[i - 1] for q in net.pos[i]]
        pairs = pairs[::max(1, int(np.ceil(len(pairs) / 60)))]
        scene.play(*[lit(d) for d in net.nodes[i - 1]],
                   *[ShowPassingFlash(Line(a, b, color=ACCENT, stroke_width=3.5), time_width=0.8) for a, b in pairs],
                   run_time=step)


class ResBlock3D:
    def __init__(self, at=ORIGIN, gap=1.8):
        at = np.array(at, dtype=float)
        xs = [(k - 1.5) * gap for k in range(4)]
        self.P = [at + np.array([x, 0, 0]) for x in xs]
        self.at = at
        self.inp = Dot3D(self.P[0], radius=0.16, color=K)
        self.l1 = box((0.5, 1.0, 1.0), self.P[1]); self.l2 = box((0.5, 1.0, 1.0), self.P[2])
        self.add_node = sphere(0.3, self.P[3])
        R = (xs[3] - xs[0]) / 2; cx = at + np.array([(xs[0] + xs[3]) / 2, 0, 0])
        self.skip = curve3d(lambda t: [cx[0] - R * np.cos(t), cx[1], cx[2] + 1.6 * np.sin(t)], (0, PI), K, 4)
        dx = np.array([0.25, 0, 0])
        self.segs = [(self.P[0], self.P[1] - dx), (self.P[1] + dx, self.P[2] - dx),
                     (self.P[2] + dx, self.P[3] - np.array([0.3, 0, 0]))]
        self.lines = VGroup(*[Line(a, b, color=K, stroke_width=3) for a, b in self.segs])
        self.group = VGroup(self.inp, self.l1, self.l2, self.add_node, self.lines, self.skip)

    def reveal(self, scene, run=1.6):
        scene.play(FadeIn(self.inp), FadeIn(self.l1), FadeIn(self.l2), FadeIn(self.add_node),
                   Create(self.lines), Create(self.skip), run_time=run)
        scene.keep(self.group)
        tags(scene, [("$x$", self.P[0] + np.array([0, 0, -0.5])), ("$F(x)$", self.P[1] + np.array([0.9, 0, 0.9])),
                     ("$x+F(x)$", self.P[3] + np.array([0, 0, 0.6])), ("skip", self.at + np.array([0, 0, 2.0]))], 24)

    def fire(self, scene, step=0.6):
        scene.play(lit(self.inp), run_time=step * 0.4)
        flashes(scene, [self.segs[0]], run=step); scene.play(lit(self.l1), run_time=step * 0.4)
        flashes(scene, [self.segs[1]], run=step); scene.play(lit(self.l2), run_time=step * 0.4)
        flashes(scene, [self.segs[2]], run=step)
        scene.play(ShowPassingFlash(self.skip.copy().set_stroke(ACCENT, width=6), time_width=0.6), run_time=step * 1.6)
        scene.play(lit(self.add_node), run_time=step * 0.4)

    def reset(self, scene, run=0.5):
        scene.play(*[unlit(m) for m in (self.inp, self.l1, self.l2, self.add_node)], run_time=run)
