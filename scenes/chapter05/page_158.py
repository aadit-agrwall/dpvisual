import json
import os
import pathlib
import subprocess
import sys
from collections import Counter
import numpy as np
from manim import *
from lib3d import *
from lib3d.ml.basics_fit import _stage_floor
from scenes.chapter05.page_143 import SR, write_wav
from scenes.chapter05.page_145 import Page145, build_audio, _cloud, _forget, ZOOM
import os, sys, subprocess, json
import numpy as np
from manim import *
from lib3d import *
from scenes.chapter05.page_146 import Page146

# --- runner deps ---
import pathlib
from collections import Counter
from scenes.chapter05.page_145 import build_audio
from scenes.chapter05.page_143 import write_wav
ROOT = pathlib.Path(__file__).resolve().parents[2]
# --- end runner deps ---


def _blk(pos, s=(1.4, 1.0, 0.6)):
    m = Prism(dimensions=list(s))
    m.set_fill("#FFFFFF", 1); m.set_stroke("#000000", 2)
    m.move_to(np.array(pos, dtype=float))
    return m


def _wire(f, rng=2.6, n=7, m=8):
    out = []
    us = np.linspace(-rng, rng, n)
    vs = np.linspace(-rng, rng, m + 1)
    for u in us:
        for a, b in zip(vs[:-1], vs[1:]):
            out.append(segment(np.array([u, a, f(u, a)]), np.array([u, b, f(u, b)]), width=2))
    for v in us:
        for a, b in zip(vs[:-1], vs[1:]):
            out.append(segment(np.array([a, v, f(a, v)]), np.array([b, v, f(b, v)]), width=2))
    return out


def _curve_segs(fn, t0, t1, n=40, width=4):
    ts = np.linspace(t0, t1, n)
    P = [np.array(fn(t), dtype=float) for t in ts]
    return [segment(a, b, width=width) for a, b in zip(P[:-1], P[1:])], P


def _twist(t):
    return (2.6 * np.cos(t) + 0.6 * np.cos(2.2 * t), 1.6 * np.sin(t) * np.cos(0.5 * t), 0.5 * np.sin(1.5 * t))


def _eight(t):
    return (2.6 * np.sin(t), 1.5 * np.sin(2 * t), 0.0)


class Page158(Page146):

    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_string(); self.wipe()
        self.s_dir(); self.wipe()
        self.s_eight(); self.wipe()
        self.s_invalid(); self.wipe()
        self.s_between(); self.wipe()
        self.wait(0.4)

    # ---- 1. Figure 5.11 ----
    def s_string(self):
        self.caption(r"Figure 5.11: data near a one-dimensional manifold")
        self.say(r"Two-dimensional space, but the data is concentrated near a twisted string")
        self.go("oblique"); self.floor(-0.9)
        segs, P = _curve_segs(_twist, 0.0, 2 * np.pi * 0.95, n=48)
        lag = 0.02
        self.act("stagger", LaggedStart(*[Create(s) for s in segs], lag_ratio=lag),
                 run=1.4, n=len(segs), lag=lag, at=0.0)
        self.keep(*segs)
        rng = np.random.RandomState(9)
        pts = []
        for _ in range(70):
            p = P[rng.randint(len(P))]
            pts.append(Dot3D(p + rng.normal(0, 0.14, 3) * np.array([1, 1, 0.5]), radius=0.07, color=K))
        lag2 = 0.012
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in pts], lag_ratio=lag2),
                 run=1.5, n=len(pts), lag=lag2, at=0.0)
        self.keep(*pts)
        self.say(r"The solid line is the manifold the learner should infer")
        self.act("zap", *[s.animate.set_color(ACCENT) for s in segs], run=0.8)
        self.orbit(0.35, run=1.6)

    # ---- 2. local direction of variation ----
    def s_dir(self):
        self.caption(r"Each dimension is a local direction of variation")
        self.say(r"A manifold embedded in a higher-dimensional space")
        self.go("oblique"); self.floor(-0.9)
        segs, P = _curve_segs(_twist, 0.0, 2 * np.pi * 0.95, n=48)
        self.act("cut", *[Create(s) for s in segs], run=1.2)
        self.keep(*segs)
        k = 12
        dot = Dot3D(P[k] + np.array([0.0, 0.0, 0.1]), radius=0.15, color=ACCENT)
        self.act("pop", FadeIn(dot), run=0.5)
        self.keep(dot)
        d = P[k + 1] - P[k - 1]
        d = d / np.linalg.norm(d)
        ar = arrow3d(P[k] + np.array([0.0, 0.0, 0.1]) - 1.0 * d, P[k] + np.array([0.0, 0.0, 0.1]) + 1.0 * d, color=ACCENT)
        self.act("zap", FadeIn(ar), run=0.7)
        self.keep(ar)
        self.say(r"Each dimension corresponds to a direction along which the data can vary")
        self.orbit(0.35, run=1.6)

    # ---- 3. figure eight ----
    def s_eight(self):
        self.caption(r"A figure eight: the dimensionality varies")
        self.say(r"A single dimension in most places")
        self.go("oblique"); self.floor(-0.5)
        segs, P = _curve_segs(_eight, 0.0, 2 * np.pi, n=60, width=5)
        lag = 0.015
        self.act("stagger", LaggedStart(*[Create(s) for s in segs], lag_ratio=lag),
                 run=1.4, n=len(segs), lag=lag, at=0.0)
        self.keep(*segs)
        dot = Dot3D(P[5] + np.array([0.0, 0.0, 0.1]), radius=0.15, color=ACCENT)
        self.act("pop", FadeIn(dot), run=0.5)
        self.keep(dot)
        for i in (12, 20, 28):
            self.act("cut", dot.animate.move_to(P[i] + np.array([0.0, 0.0, 0.1])), run=0.5)
        self.say(r"But two dimensions at the intersection in the center")
        self.act("cut", dot.animate.move_to(np.array([0.0, 0.0, 0.1])), run=0.6)
        patch = _blk((0.0, 0.0, 0.05), (1.0, 1.0, 0.05))
        patch.set_fill(ACCENT, 0.5)
        self.act("pop", FadeIn(patch), run=0.6)
        self.keep(patch)
        self.orbit(0.5, run=2.0)

    # ---- 4. most of R^n is invalid ----
    def s_invalid(self):
        self.caption(r"Most of $\mathbb{R}^{n}$ consists of invalid inputs")
        self.say(r"Interesting inputs occur only along a collection of manifolds")
        self.go("oblique"); self.floor(-0.4)
        wire = _wire(lambda u, v: 0.0, rng=3.0, n=8, m=10)
        lag = 0.004
        self.act("stagger", LaggedStart(*[FadeIn(w) for w in wire], lag_ratio=lag),
                 run=1.3, n=len(wire), lag=lag, at=0.0)
        self.keep(*wire)
        self.say(r"A small subset of points holds most of the probability mass")
        ts = np.linspace(-2.6, 2.6, 24)
        rib = [Dot3D(np.array([t, 0.8 * np.sin(1.1 * t), 0.1]), radius=0.1, color=ACCENT) for t in ts]
        lag2 = 0.04
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in rib], lag_ratio=lag2),
                 run=1.4, n=len(rib), lag=lag2, at=0.0)
        self.keep(*rib)
        self.orbit(0.35, run=1.6)

    # ---- 5. variation along or between manifolds ----
    def s_between(self):
        self.caption(r"Interesting variation lies on the manifold or between manifolds")
        self.say(r"Output variations occur only along directions on the manifold")
        self.go("oblique"); self.floor(-0.9)
        fa = lambda t: (t, 1.2 * np.sin(0.9 * t) + 1.3, 0.0)
        fb = lambda t: (t, -1.2 * np.sin(0.9 * t) - 1.3, 0.0)
        sa, Pa = _curve_segs(fa, -2.8, 2.8, n=30, width=5)
        sb, Pb = _curve_segs(fb, -2.8, 2.8, n=30, width=5)
        self.act("cut", *[Create(s) for s in sa + sb], run=1.2)
        self.keep(*sa, *sb)
        dot = Dot3D(Pa[2] + np.array([0.0, 0.0, 0.1]), radius=0.15, color=ACCENT)
        self.act("pop", FadeIn(dot), run=0.5)
        self.keep(dot)
        for i in (8, 15, 22):
            self.act("cut", dot.animate.move_to(Pa[i] + np.array([0.0, 0.0, 0.1])), run=0.5)
        self.say(r"Or when we move from one manifold to another")
        self.act("zap", dot.animate.move_to(Pb[22] + np.array([0.0, 0.0, 0.1])), run=0.9)
        self.orbit(0.5, run=2.0)


def main():
    os.chdir(ROOT)
    res = {"preview": "540,540", "reel": "1080,1080", "youtube": "2560,1440", "sq1440": "1440,1440"}.get(os.environ.get("P158_MODE", "reel"), "1080,1080")
    name = "Page158_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page158_timeline.json"
    wav = ROOT / "media/sfx/Page158.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_158.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page158"],
                            env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not tl.exists():
        print("RENDER FAILED. last lines of", log)
        print("".join(log.read_text().splitlines(True)[-25:]))
        sys.exit(1)
    vids = [v for v in (ROOT / "media/videos").rglob(name + ".mp4") if "partial_movie_files" not in v.parts]
    vid = sorted(vids, key=lambda p: p.stat().st_mtime)[-1]
    data = json.load(open(tl))
    buf, slog = build_audio(data)
    write_wav(buf, wav)
    final = out_dir / (name + "_sfx.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(vid), "-i", str(wav), "-map", "0:v",
                    "-map", "1:a", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    "-movflags", "+faststart", str(final)], check=True)
    sheet = out_dir / (name + "_sheet.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(final), "-vf",
                    "fps=1/4,scale=360:-1,tile=4x4", "-frames:v", "1", str(sheet)])
    skipped = sum(1 for e in data["events"] if "Camera" in e["kinds"] and not e["vis"])
    print("=== SFX REPORT ===")
    print(f"timeline {data['total']:.2f}s | {len(data['events'])} plays | {len(data['cues'])} cues | {len(slog)} sounds")
    print("sounds:", dict(Counter(n for _, n in slog)))
    print(f"camera moves with an empty stage (silenced): {skipped}")
    try:
        vd = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                      "-of", "csv=p=0", str(vid)], text=True).strip()
        print(f"video duration {vd}s vs timeline {data['total']:.2f}s (should match within ~0.1s)")
    except Exception:
        pass
    print("SAVED:", final)
    print("CHECK SHEET:", sheet)
    if not os.environ.get("NOOPEN"): subprocess.run(["open", str(final)])
    if not os.environ.get("NOOPEN"): subprocess.run(["open", str(sheet)])


if __name__ == "__main__":
    main()
