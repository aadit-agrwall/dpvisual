"""Sequence models: RNN, LSTM, attention."""
from manim import *
import numpy as np
from lib3d.core.engine3d import *
from lib3d.ml.helpers import *
from lib3d.ml.helpers import _u


class RNNUnrolled3D:
    def __init__(self, steps=4, gap=2.0, at=ORIGIN):
        at = np.array(at, dtype=float); self.steps = steps
        self.cp = [at + np.array([(t - (steps - 1) / 2) * gap, 0, 0]) for t in range(steps)]
        self.ip = [p + np.array([0, 0, -1.6]) for p in self.cp]
        self.op = [p + np.array([0, 0, 1.6]) for p in self.cp]
        z, x = np.array([0, 0, 0.45]), np.array([0.45, 0, 0])
        self.cells = VGroup(*[box((0.9, 0.9, 0.9), p) for p in self.cp])
        self.ins = VGroup(*[Dot3D(p, radius=0.15, color=K) for p in self.ip])
        self.outs = VGroup(*[Dot3D(p, radius=0.15, color=K) for p in self.op])
        self.ie = [(self.ip[t], self.cp[t] - z) for t in range(steps)]
        self.oe = [(self.cp[t] + z, self.op[t]) for t in range(steps)]
        self.he = [(self.cp[t] + x, self.cp[t + 1] - x) for t in range(steps - 1)]
        lines = [Line(a, b, color=K, stroke_width=3) for a, b in self.ie + self.oe + self.he]
        self.lines = VGroup(*lines)
        self.group = VGroup(self.cells, self.ins, self.outs, self.lines)

    def reveal(self, scene, run=2.0):
        scene.play(LaggedStart(*[FadeIn(c) for c in self.cells], lag_ratio=0.25), run_time=run * 0.5)
        scene.play(FadeIn(self.ins), FadeIn(self.outs), Create(self.lines), run_time=run * 0.5)
        scene.keep(self.group)
        items = []
        for t in range(self.steps):
            items.append((f"$x_{{{t + 1}}}$", self.ip[t] + np.array([0, 0, -0.45])))
            items.append((f"$y_{{{t + 1}}}$", self.op[t] + np.array([0, 0, 0.45])))
        tags(scene, items, 24)

    def run(self, scene, step=0.5):
        for t in range(self.steps):
            scene.play(lit(self.ins[t]), run_time=step * 0.4)
            pairs = [self.ie[t]] + ([self.he[t - 1]] if t > 0 else [])
            flashes(scene, pairs, run=step)
            scene.play(lit(self.cells[t]), run_time=step * 0.4)
            flashes(scene, [self.oe[t]], run=step * 0.8)
            scene.play(lit(self.outs[t]), run_time=step * 0.4)

    def reset(self, scene, run=0.5):
        scene.play(*[unlit(m) for m in list(self.ins) + list(self.cells) + list(self.outs)], run_time=run)


class LSTMCell3D:
    def __init__(self, at=ORIGIN):
        at = np.array(at, dtype=float)
        A = lambda x, z: at + np.array([x, 0.0, z])
        self.A = A
        self.frame = Polygon(A(-2.2, -1.1), A(2.2, -1.1), A(2.2, 1.1), A(-2.2, 1.1), color=K, stroke_width=3)
        self.frame.set_fill(PAPER, opacity=0.05)
        self.gx = {"f": -1.3, "i": -0.3, "o": 1.2}
        self.c_pair = (A(-2.9, 0.7), A(2.9, 0.7))
        self.c_line = Line(*self.c_pair, color=K, stroke_width=4)
        self.gates = {k: box((0.42, 0.42, 0.42), A(x, -0.2), sw=2) for k, x in self.gx.items()}
        self.x_in = Dot3D(A(-0.2, -1.9), radius=0.15, color=K)
        self.c_out = Dot3D(A(2.9, 0.7), radius=0.15, color=K)
        self.h_out = Dot3D(A(2.9, -0.2), radius=0.15, color=K)
        self.x_pairs = [(A(-0.2, -1.9), A(x, -0.42)) for x in self.gx.values()]
        self.g_pairs = [(A(self.gx["f"], 0.02), A(self.gx["f"], 0.7)), (A(self.gx["i"], 0.02), A(self.gx["i"], 0.7))]
        self.o_pair = (A(self.gx["o"] + 0.21, -0.2), A(2.9, -0.2))
        lines = [Line(a, b, color=K, stroke_width=2.5) for a, b in self.x_pairs + self.g_pairs + [self.o_pair]]
        self.lines = VGroup(*lines)
        self.group = VGroup(self.frame, self.c_line, self.lines, *self.gates.values(),
                            self.x_in, self.c_out, self.h_out)

    def reveal(self, scene, run=1.6):
        scene.play(Create(self.frame), Create(self.c_line), run_time=run * 0.4)
        scene.play(*[FadeIn(g) for g in self.gates.values()], FadeIn(self.x_in), FadeIn(self.c_out),
                   FadeIn(self.h_out), Create(self.lines), run_time=run * 0.6)
        scene.keep(self.group)
        A = self.A
        tags(scene, [("$f_t$", A(self.gx["f"] + 0.35, 0.25)), ("$i_t$", A(self.gx["i"] + 0.35, 0.25)),
                     ("$o_t$", A(self.gx["o"], 0.35)), ("$c_t$", A(2.9, 1.05)),
                     ("$x_t$", A(-0.2, -2.3)), ("$h_t$", A(2.9, -0.6))], 24)

    def fire(self, scene, step=0.5):
        scene.play(lit(self.x_in), run_time=step * 0.4)
        flashes(scene, self.x_pairs, run=step)
        scene.play(*[lit(g) for g in self.gates.values()], run_time=step * 0.4)
        flashes(scene, self.g_pairs + [self.c_pair], run=step * 1.4)
        scene.play(lit(self.c_out), run_time=step * 0.4)
        flashes(scene, [self.o_pair], run=step)
        scene.play(lit(self.h_out), run_time=step * 0.4)

    def reset(self, scene, run=0.5):
        parts = list(self.gates.values()) + [self.x_in, self.c_out, self.h_out]
        scene.play(*[unlit(m) for m in parts], run_time=run)


class Attention3D:
    def __init__(self, n=5, cell=0.45, gap=0.1, seed=1, sharp=2.2, at=ORIGIN):
        at = np.array(at, dtype=float)
        rng = np.random.default_rng(seed)
        S = rng.normal(0, 1, (n, n)) + sharp * np.eye(n)
        E = np.exp(S); self.W = E / E.sum(axis=1, keepdims=True)
        self.n, self.at, self.span = n, at, n * (cell + gap) / 2
        self.bars = bars3d(self.W, cell=cell, gap=gap, height_scale=3.0, at=at)
        self.group = VGroup(self.bars)

    def reveal(self, scene, run=1.6):
        scene.play(LaggedStart(*[FadeIn(b) for b in self.bars], lag_ratio=0.04), run_time=run)
        scene.keep(self.group)
        tags(scene, [("queries", self.at + np.array([-(self.span + 0.7), 0, 0.3])),
                     ("keys", self.at + np.array([0, -(self.span + 0.5), 0.0]))])

    def focus(self, scene, q, run=0.6):
        scene.play(*[lit(self.bars[q * self.n + j]) for j in range(self.n)], run_time=run)

    def unfocus(self, scene, q, run=0.4):
        scene.play(*[unlit(self.bars[q * self.n + j]) for j in range(self.n)], run_time=run)

    def sweep(self, scene, rows=None, run=0.6):
        for q in (range(self.n) if rows is None else rows):
            self.focus(scene, q, run); self.unfocus(scene, q)
