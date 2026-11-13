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


def _gray(rng):
    g = int(rng.uniform(60, 235))
    return "#%02X%02X%02X" % (g, g, g)


def _curve_segs(fn, t0, t1, n=40, width=4):
    ts = np.linspace(t0, t1, n)
    P = [np.array(fn(t), dtype=float) for t in ts]
    return [segment(a, b, width=width) for a, b in zip(P[:-1], P[1:])], P


def _road(t):
    return (t, 1.3 * np.sin(0.9 * t), 0.4 * np.cos(0.7 * t))


class Page160(Page146):

    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_conc(); self.wipe()
        self.s_trans(); self.wipe()
        self.s_multi(); self.wipe()
        self.s_road(); self.wipe()
        self.s_coord(); self.wipe()
        self.wait(0.4)

    # ---- 1. concentrated probability ----
    def s_conc(self):
        self.caption(r"Probability over real-world data is highly concentrated")
        self.say(r"Uniform noise essentially never resembles structured inputs")
        self.go("oblique"); self.floor(-0.1)
        rng = np.random.RandomState(4)
        n, st = 8, 0.5
        cells = []
        for i in range(n):
            for j in range(n):
                p = np.array([(i - 3.5) * st, (j - 1.0) * st + 0.9, 0.0])
                b = _blk(p, (st * 0.9, st * 0.9, 0.06))
                b.set_fill(_gray(rng), 1)
                cells.append(b)
        lag = 0.01
        self.act("stagger", LaggedStart(*[FadeIn(b) for b in cells], lag_ratio=lag),
                 run=1.3, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        self.say(r"Random letters almost never give meaningful English text")
        row = [_blk(np.array([(i - 5.5) * 0.62, -2.4, 0.1]), (0.5, 0.5, 0.2)) for i in range(12)]
        lag2 = 0.05
        self.act("stagger", LaggedStart(*[FadeIn(b) for b in row], lag_ratio=lag2),
                 run=1.0, n=len(row), lag=lag2, at=0.0)
        self.keep(*row)
        self.act("zap", *[row[i].animate.set_fill(ACCENT, 0.9) for i in (1, 6)], run=0.6)
        self.orbit(0.35, run=1.6)

    # ---- 2. transformations trace out a manifold ----
    def s_trans(self):
        self.caption(r"Examples are connected by transformations")
        self.say(r"Gradually dim or brighten the lights, move or rotate objects")
        self.go("oblique"); self.floor(-0.1)
        k = 8
        xs = [(-3.5 + i * 1.0) for i in range(k)]
        tiles = []
        for i, x in enumerate(xs):
            g = int(250 - i * 25)
            b = _blk(np.array([x, 0.0, 0.05]), (0.8, 0.8, 0.1))
            b.set_fill("#%02X%02X%02X" % (g, g, g), 1)
            tiles.append(b)
        lag = 0.08
        self.act("stagger", LaggedStart(*[FadeIn(b) for b in tiles], lag_ratio=lag),
                 run=1.2, n=len(tiles), lag=lag, at=0.0)
        self.keep(*tiles)
        links = [segment(np.array([a, 0.0, 0.2]), np.array([b, 0.0, 0.2]), width=4) for a, b in zip(xs[:-1], xs[1:])]
        self.act("cut", *[Create(l) for l in links], run=0.8)
        self.keep(*links)
        self.say(r"Each example is surrounded by highly similar examples")
        dot = Dot3D(np.array([xs[0], 0.0, 0.5]), radius=0.15, color=ACCENT)
        self.act("pop", FadeIn(dot), run=0.5)
        self.keep(dot)
        for x in xs[1:]:
            self.act("cut", dot.animate.move_to(np.array([x, 0.0, 0.5])), run=0.35)
        self.orbit(0.35, run=1.6)

    # ---- 3. multiple manifolds ----
    def s_multi(self):
        self.caption(r"Multiple manifolds are likely involved")
        self.say(r"The manifold of human faces may not connect to the manifold of cat faces")
        self.go("oblique"); self.floor(-0.9)
        fa = lambda t: (t, 1.2 * np.sin(0.9 * t) + 1.5, 0.0)
        fb = lambda t: (t, -1.2 * np.sin(0.9 * t) - 1.5, 0.0)
        sa, Pa = _curve_segs(fa, -2.8, 2.8, n=26, width=4)
        sb, Pb = _curve_segs(fb, -2.8, 2.8, n=26, width=4)
        self.act("cut", *[Create(s) for s in sa + sb], run=1.2)
        self.keep(*sa, *sb)
        rng = np.random.RandomState(6)
        da = [Dot3D(Pa[rng.randint(len(Pa))] + rng.normal(0, 0.1, 3) * np.array([1, 1, 0]), radius=0.08, color=ACCENT)
              for _ in range(18)]
        db = [Dot3D(Pb[rng.randint(len(Pb))] + rng.normal(0, 0.1, 3) * np.array([1, 1, 0]), radius=0.08, color=K)
              for _ in range(18)]
        lag = 0.03
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in da + db], lag_ratio=lag),
                 run=1.4, n=len(da) + len(db), lag=lag, at=0.0)
        self.keep(*da, *db)
        self.say(r"Two separate manifolds, with a gap between them")
        self.orbit(0.5, run=2.0)

    # ---- 4. roads as 1-D manifolds in 3-D ----
    def s_road(self):
        self.caption(r"Roads are 1-D manifolds embedded in 3-D space")
        self.say(r"Data on a low-dimensional manifold")
        self.go("oblique"); self.floor(-1.2)
        segs, P = _curve_segs(_road, -3.0, 3.0, n=40, width=6)
        lag = 0.02
        self.act("stagger", LaggedStart(*[Create(s) for s in segs], lag_ratio=lag),
                 run=1.3, n=len(segs), lag=lag, at=0.0)
        self.keep(*segs)
        houses = [_blk(P[i] + np.array([0.0, 0.7 if i % 2 else -0.7, 0.2]), (0.4, 0.4, 0.4)) for i in range(3, 38, 5)]
        self.act("pop", *[FadeIn(h) for h in houses], run=0.8)
        self.keep(*houses)
        self.say(r"Addresses lie along the road")
        dot = Dot3D(P[0] + np.array([0.0, 0.0, 0.15]), radius=0.15, color=ACCENT)
        self.act("pop", FadeIn(dot), run=0.5)
        self.keep(dot)
        for i in (10, 20, 30):
            self.act("cut", dot.animate.move_to(P[i] + np.array([0.0, 0.0, 0.15])), run=0.5)
        self.orbit(0.35, run=1.6)

    # ---- 5. manifold coordinates ----
    def s_coord(self):
        self.caption(r"Represent data by coordinates on the manifold")
        self.say(r"Address numbers along the road, not coordinates in 3-D space")
        self.go("oblique"); self.floor(-1.2)
        segs, P = _curve_segs(_road, -3.0, 3.0, n=40, width=6)
        self.act("cut", *[Create(s) for s in segs], run=1.0)
        self.keep(*segs)
        line = segment(np.array([-3.0, -2.6, 0.0]), np.array([3.0, -2.6, 0.0]), width=6)
        ticks = [segment(np.array([x, -2.6, 0.0]), np.array([x, -2.6, 0.35]), width=4)
                 for x in np.linspace(-3.0, 3.0, 7)]
        self.act("pop", Create(line), *[Create(t) for t in ticks], run=0.9)
        self.keep(line, *ticks)
        self.say(r"Extracting these coordinates is challenging, but promises better algorithms")
        dot = Dot3D(P[20] + np.array([0.0, 0.0, 0.15]), radius=0.15, color=ACCENT)
        self.act("pop", FadeIn(dot), run=0.5)
        self.keep(dot)
        self.act("zap", dot.animate.move_to(np.array([P[20][0], -2.6, 0.15])), run=1.0)
        self.act("zap", ticks[3].animate.set_color(ACCENT), run=0.5)
        self.orbit(0.5, run=2.0)


def main():
    os.chdir(ROOT)
    res = {"preview": "540,540", "reel": "1080,1080", "youtube": "2560,1440", "sq1440": "1440,1440"}.get(os.environ.get("P160_MODE", "reel"), "1080,1080")
    name = "Page160_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page160_timeline.json"
    wav = ROOT / "media/sfx/Page160.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_160.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page160"],
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
