"""Shared helpers: lit/unlit, tags, flashes."""
from manim import *
import numpy as np
from lib3d.core.engine3d import *


def _u(v):
    v = np.array(v, dtype=float); n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 0.0, 1.0])


def lit(m, op=0.85):
    """Blue 'active' state. Dots recolor; boxes/spheres/squares fill blue."""
    if isinstance(m, Dot3D):
        return m.animate.set_color(ACCENT)
    return m.animate.set_fill(ACCENT, opacity=op)


def unlit(m):
    if isinstance(m, Dot3D):
        return m.animate.set_color(K)
    return m.animate.set_fill(PAPER, opacity=0.9)


def tags(scene, items, size=24, run=0.4):
    """Flat text labels that stay readable in every camera view."""
    out = []
    for text, pos in items:
        t = Tex(text, color=K, font_size=size).move_to(np.array(pos, dtype=float))
        scene.add_fixed_orientation_mobjects(t)
        out.append(t)
    scene._items().extend(out)
    if out:
        scene.play(*[FadeIn(t) for t in out], run_time=run)
    return out


def flashes(scene, pairs, run=0.6, width=5):
    if not pairs:
        return
    scene.play(*[ShowPassingFlash(Line(np.array(a, dtype=float), np.array(b, dtype=float),
                                       color=ACCENT, stroke_width=width), time_width=0.7)
                 for a, b in pairs], run_time=run)


def _perc_reset(self, scene, run=0.5):
    parts = list(self.inputs) + [self.sum_node, self.out_node]
    scene.play(*[unlit(m) for m in parts], run_time=run)


Perceptron3D.reset = _perc_reset   # fixes boxes turning black after reset
