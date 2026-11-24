import os, json
import numpy as np
from manim import *

BG = __import__("os").environ.get("TOM_BG", "#002A66")
INK = "#FFFFFF"; HEAD = "#FFDE21"
GRN = "#7CFFB2"; BLU = "#CFE3FF"; DIM = "#A9C4F0"; RED = "#FF9A9A"; ACC = "#FFDE21"
PHI = 68 * DEGREES; THETA = -35 * DEGREES

def setup(page):
    mode = os.environ.get(f"P{page}_MODE", "preview")
    w, h, fps = {"sq1440": (1440, 1440, 30), "yt1080": (1240, 1080, 30),
                 "portrait": (1080, 1920, 30), "landscape": (1920, 1080, 30)}.get(mode, (720, 720, 15))
    config.pixel_width = w; config.pixel_height = h; config.frame_rate = fps
    config.frame_height = 8 * max(1, h / w)
    config.frame_width = config.frame_height * w / h
    config.background_color = BG

def T(s, size=26, color=INK, maxw=7.2):
    t = Text(s, font_size=size, color=color)
    if t.width > maxw: t.scale_to_fit_width(maxw)
    return t

def H(s, size=34, maxw=7.4):
    t = Text(s, font_size=size, color=HEAD, weight=BOLD)
    if t.width > maxw: t.scale_to_fit_width(maxw)
    return t

def cap(s, size=24):
    return T(s, size, INK, 5.6).move_to(DOWN * 3.3 + LEFT * 1.1)

def ball(p, r=0.17, color=BLU, opacity=1.0):
    return Sphere(center=p, radius=r, resolution=(10, 10), fill_opacity=opacity,
                  checkerboard_colors=False, stroke_width=0).set_color(color)

def net3d(sizes, x0=-2.3, x1=2.3, z0=0.0, color=BLU, gap=0.62):
    pts = []
    for i, n in enumerate(sizes):
        x = x0 + (x1 - x0) * i / (len(sizes) - 1)
        pts.append([np.array([x, ((j % 2) - 0.5) * 0.9, z0 + (j - (n - 1) / 2) * gap]) for j in range(n)])
    edges = VGroup()
    for a, b in zip(pts[:-1], pts[1:]):
        for p in a:
            for q in b:
                edges.add(Line(p, q, stroke_width=1.2, stroke_opacity=0.4, color=color))
    layers = VGroup(*[VGroup(*[ball(p, color=color) for p in col]) for col in pts])
    return edges, layers

def vec3d(n, color):
    cs = [Cube(side_length=0.3, fill_opacity=0.9, fill_color=color, stroke_width=1, stroke_color=INK) for _ in range(n)]
    return VGroup(*cs).arrange(OUT, buff=0.05)

def slab(w, h, x, zbase, color):
    return Prism(dimensions=[w, w, h], fill_opacity=0.9, fill_color=color, stroke_width=0.5).move_to([x, 0, zbase + h / 2])

def surf(fn, rng=2.0, color=BLU, opacity=0.8, res=(16, 16)):
    s = Surface(lambda u, v: np.array([u, v, fn(u, v)]), u_range=[-rng, rng], v_range=[-rng, rng],
                resolution=res, fill_opacity=opacity, checkerboard_colors=False,
                stroke_width=0.4, stroke_color=INK)
    return s.set_fill(color, opacity=opacity)

class TomScene(ThreeDScene):
    def spin(self):
        self.stop_ambient_camera_rotation()
        self.set_camera_orientation(phi=PHI, theta=THETA)
        self.begin_ambient_camera_rotation(rate=0.04)
    def flat_view(self):
        self.stop_ambient_camera_rotation()
        self.set_camera_orientation(phi=PHI, theta=-90 * DEGREES)
    def begin(self):
        self.spin()
        self.seg = self.renderer.time
        self.cuts = []
    def hud(self, *m):
        self.add_fixed_in_frame_mobjects(*m)
    def end_scene(self, total, last=False):
        used = self.renderer.time - self.seg
        if used < total - 0.05:
            self.wait(total - used)
        else:
            print(f"WARN: scene ran {used:.1f}s, longer than {total}s")
        if not last:
            items = [m for m in self.mobjects if not isinstance(m, ValueTracker)]
            self.play(*[FadeOut(m) for m in items], run_time=0.5)
            self.spin()
            self.cuts.append(round(self.renderer.time, 2))
        self.seg = self.renderer.time
    def save_cues(self, page):
        os.makedirs("media/sfx", exist_ok=True)
        json.dump([{"type": "cut", "t": t} for t in self.cuts], open(f"media/sfx/page_{page}_cuts.json", "w"))
