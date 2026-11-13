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


class Page156(Page146):

    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_cover(); self.wipe()
        self.s_peaks(); self.wipe()
        self.s_highdim(); self.wipe()
        self.s_voronoi(); self.wipe()
        self.s_exp(); self.wipe()
        self.wait(0.4)

    # ---- 1. cover each cell with at least one example ----
    def s_cover(self):
        self.caption(r"Smoothness alone: every cell needs an example")
        self.say(r"The only way to get the whole checkerboard right is to cover each cell")
        self.go("oblique"); self.floor(-0.1)
        n, step = 6, 0.8
        cells, cen = [], []
        for i in range(n):
            for j in range(n):
                p = np.array([(i - 2.5) * step, (j - 2.5) * step])
                cells.append(_blk((p[0], p[1], 0.05), (0.72, 0.72, 0.1)))
                cen.append(p)
        lag = 0.015
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.2, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        dots = [Dot3D(np.array([p[0], p[1], 0.22]), radius=0.1, color=ACCENT) for p in cen]
        lag2 = 0.02
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag2),
                 run=1.4, n=len(dots), lag=lag2, at=0.0)
        self.keep(*dots)
        self.say(r"At least one example in each cell")
        self.act("zap", *[c.animate.set_fill(ACCENT, 0.35) for c in cells], run=0.7)
        self.orbit(0.35, run=1.6)

    # ---- 2. peaks and valleys ----
    def s_peaks(self):
        self.caption(r"Nonparametric learning needs to see peaks and valleys")
        self.say(r"Enough examples to observe high points on most peaks")
        self.go("oblique"); self.floor(-1.2)
        f = lambda u, v: 0.7 * np.sin(1.1 * u) * np.cos(1.1 * v)
        wire = _wire(f, rng=2.8, n=9, m=12)
        lag = 0.003
        self.act("stagger", LaggedStart(*[FadeIn(w) for w in wire], lag_ratio=lag),
                 run=1.4, n=len(wire), lag=lag, at=0.0)
        self.keep(*wire)
        pk = [(np.pi / 2.2, 0.0), (-np.pi / 2.2, 0.0)]
        vl = [(np.pi / 2.2, np.pi / 1.1), (-np.pi / 2.2, np.pi / 1.1)]
        hi = [Dot3D(np.array([u, v, f(u, v) + 0.1]), radius=0.13, color=ACCENT) for u, v in pk]
        self.act("pop", *[FadeIn(d) for d in hi], run=0.6)
        self.keep(*hi)
        self.say(r"And low points on most valleys")
        lo = [Dot3D(np.array([u, v, f(u, v) + 0.1]), radius=0.13, color=K) for u, v in vl]
        self.act("pop", *[FadeIn(d) for d in lo], run=0.6)
        self.keep(*lo)
        self.orbit(0.35, run=1.6)

    # ---- 3. high dimensions: different change along each dimension ----
    def s_highdim(self):
        self.caption(r"In high dimensions a smooth function can still be complicated")
        self.say(r"It changes smoothly, but in a different way along each dimension")
        self.go("oblique"); self.floor(-1.4)
        f = lambda u, v: 0.5 * np.sin(1.6 * u) + 0.3 * v
        wire = _wire(f, rng=2.6, n=8, m=10)
        lag = 0.004
        self.act("stagger", LaggedStart(*[FadeIn(w) for w in wire], lag_ratio=lag),
                 run=1.4, n=len(wire), lag=lag, at=0.0)
        self.keep(*wire)
        ax = arrow3d(np.array([-2.8, -3.0, -1.0]), np.array([2.8, -3.0, -1.0]), color=ACCENT)
        ay = arrow3d(np.array([3.0, -2.8, -1.0]), np.array([3.0, 2.8, -1.0]), color=ACCENT)
        self.act("zap", FadeIn(ax), FadeIn(ay), run=0.8)
        self.keep(ax, ay)
        self.say(r"Different behavior in various regions is hard to describe with examples")
        self.orbit(0.5, run=2.0)

    # ---- 4. Figure 5.10: Voronoi ----
    def s_voronoi(self):
        self.caption(r"Figure 5.10: nearest neighbor breaks the input space into regions")
        self.say(r"An example within each region defines the region")
        self.go("oblique"); self.floor(-0.1)
        seeds = [np.array([0.0, 2.2]), np.array([-2.2, -1.2]), np.array([2.2, -1.2])]
        n, step = 11, 0.55
        cells, owner = [], []
        for i in range(n):
            for j in range(n):
                p = np.array([(i - 5) * step, (j - 5) * step])
                k = int(np.argmin([np.linalg.norm(p - s) for s in seeds]))
                cells.append(_blk((p[0], p[1], 0.04), (0.5, 0.5, 0.08)))
                owner.append(k)
        lag = 0.006
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.4, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        dots = [Dot3D(np.array([s[0], s[1], 0.2]), radius=0.16, color=K) for s in seeds]
        self.act("pop", *[FadeIn(d) for d in dots], run=0.7)
        self.keep(*dots)
        bd = [segment(np.array([0.0, 0.0, 0.08]), np.array([0.0, -3.2, 0.08]), width=6),
              segment(np.array([0.0, 0.0, 0.08]), np.array([-3.2, 1.9, 0.08]), width=6),
              segment(np.array([0.0, 0.0, 0.08]), np.array([3.2, 1.9, 0.08]), width=6)]
        self.act("cut", *[Create(b) for b in bd], run=0.9)
        self.keep(*bd)
        self.say(r"The regions form a Voronoi diagram")
        for k in range(3):
            reg = [c for c, o in zip(cells, owner) if o == k]
            self.act("zap", *[c.animate.set_fill(ACCENT, 0.6) for c in reg], run=0.4)
            self.act("zap", *[c.animate.set_fill("#FFFFFF", 1) for c in reg], run=0.2)
        self.say(r"The number of regions cannot grow faster than the number of examples")
        self.orbit(0.35, run=1.6)

    # ---- 5. O(2^k) regions from O(k) examples ----
    def s_exp(self):
        self.caption(r"The key insight: $O(2^{k})$ regions from $O(k)$ examples")
        self.say(r"A very large number of regions can be defined with few examples")
        self.go("oblique"); self.floor(-2.0)
        h = 1.8
        cells, cen = [], []
        for a in (-1, 1):
            for b in (-1, 1):
                for c in (-1, 1):
                    p = np.array([a, b, c]) * 0.95
                    cells.append(_blk(p, (1.6, 1.6, 1.6)))
                    cen.append(p)
        lag = 0.08
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.2, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        self.eq(r"O(2^{k})\ \mathrm{regions}\ \mathrm{from}\ O(k)\ \mathrm{examples}")
        self.say(r"Three splits already give eight regions")
        planes = [segment(np.array([-h, 0.0, 0.0]), np.array([h, 0.0, 0.0]), width=6),
                  segment(np.array([0.0, -h, 0.0]), np.array([0.0, h, 0.0]), width=6),
                  segment(np.array([0.0, 0.0, -h]), np.array([0.0, 0.0, h]), width=6)]
        self.act("cut", *[Create(p) for p in planes], run=0.9)
        self.keep(*planes)
        dots = [Dot3D(np.array([2.2, 0.0, 0.0]), radius=0.14, color=ACCENT),
                Dot3D(np.array([0.0, 2.2, 0.0]), radius=0.14, color=ACCENT),
                Dot3D(np.array([0.0, 0.0, 2.2]), radius=0.14, color=ACCENT)]
        self.act("pop", *[FadeIn(d) for d in dots], run=0.7)
        self.keep(*dots)
        self.act("zap", *[c.animate.set_fill(ACCENT, 0.25) for c in cells], run=0.7)
        self.orbit(0.5, run=2.0)


def main():
    os.chdir(ROOT)
    res = {"preview": "540,540", "reel": "1080,1080", "youtube": "2560,1440", "sq1440": "1440,1440"}.get(os.environ.get("P156_MODE", "reel"), "1080,1080")
    name = "Page156_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page156_timeline.json"
    wav = ROOT / "media/sfx/Page156.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_156.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page156"],
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
