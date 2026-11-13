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


def _tile(c, n, s, rng, h=0.06):
    out = []
    st = s / n
    for i in range(n):
        for j in range(n):
            p = np.array(c, dtype=float) + np.array([(i - (n - 1) / 2) * st, (j - (n - 1) / 2) * st, 0.0])
            b = _blk(p, (st * 0.92, st * 0.92, h))
            b.set_fill(_gray(rng), 1)
            out.append(b)
    return out


def _wirecube(h, width=3):
    cs = [np.array([a, b, c]) * h for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
    return [segment(p, q, width=width) for i, p in enumerate(cs) for q in cs[i + 1:]
            if np.sum(np.abs(p - q) > 1e-6) == 1]


class Page159(Page146):

    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_noise(); self.wipe()
        self.s_pixels(); self.wipe()
        self.s_face(); self.wipe()
        self.s_volume(); self.wipe()
        self.s_evidence(); self.wipe()
        self.wait(0.4)

    # ---- 1. Figure 5.12: noisy images ----
    def s_noise(self):
        self.caption(r"Figure 5.12: sampling images uniformly at random")
        self.say(r"Randomly picking each pixel gives rise to noisy images")
        self.go("oblique"); self.floor(-0.1)
        rng = np.random.RandomState(1)
        cols, rows, step = 4, 5, 1.25
        tiles = []
        for i in range(cols):
            for j in range(rows):
                c = ((i - 1.5) * step, (j - 2.0) * step, 0.0)
                tiles.extend(_tile(c, 3, 1.05, rng))
        lag = 0.003
        self.act("stagger", LaggedStart(*[FadeIn(t) for t in tiles], lag_ratio=lag),
                 run=1.6, n=len(tiles), lag=lag, at=0.0)
        self.keep(*tiles)
        self.say(r"Every sample looks like static")
        self.orbit(0.35, run=1.6)

    # ---- 2. one image, uniform pixels ----
    def s_pixels(self):
        self.caption(r"Each pixel is drawn from a uniform distribution")
        self.say(r"A single image: every pixel is an independent random value")
        self.go("oblique"); self.floor(-0.1)
        rng = np.random.RandomState(2)
        n = 7
        px = []
        for i in range(n):
            for j in range(n):
                p = np.array([(i - 3) * 0.6, (j - 3) * 0.6, 0.0])
                b = _blk(p, (0.55, 0.55, 0.06))
                b.set_fill(_gray(rng), 1)
                px.append(b)
        lag = 0.01
        self.act("stagger", LaggedStart(*[FadeIn(b) for b in px], lag_ratio=lag),
                 run=1.3, n=len(px), lag=lag, at=0.0)
        self.keep(*px)
        self.say(r"Heights show the random value chosen for each pixel")
        grow = [b.animate.stretch_to_fit_depth(0.1 + 0.6 * ((i * 7 + 3) % 5) / 4.0) for i, b in enumerate(px)]
        self.act("zap", *grow, run=0.9)
        self.orbit(0.5, run=2.0)

    # ---- 3. a face is possible but never seen ----
    def s_face(self):
        self.caption(r"A face has nonzero probability, but we never see one")
        self.say(r"Noise everywhere, structure almost never")
        self.go("oblique"); self.floor(-0.1)
        rng = np.random.RandomState(3)
        n, st = 10, 0.5
        cells, cen = [], []
        for i in range(n):
            for j in range(n):
                p = np.array([(i - 4.5) * st, (j - 4.5) * st, 0.0])
                b = _blk(p, (st * 0.9, st * 0.9, 0.06))
                b.set_fill(_gray(rng), 1)
                cells.append(b); cen.append(p)
        lag = 0.008
        self.act("stagger", LaggedStart(*[FadeIn(b) for b in cells], lag_ratio=lag),
                 run=1.4, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        ring = [i for i in range(n * n)
                if 1.3 < np.linalg.norm(cen[i]) < 1.9]
        self.say(r"A smooth, face-like pattern would be one very rare image")
        self.act("zap", *[cells[i].animate.set_fill(ACCENT, 0.9) for i in ring], run=0.9)
        ar = arrow3d(np.array([-3.4, -3.2, 0.3]), np.array([-1.4, -1.6, 0.3]), color=ACCENT)
        self.act("cut", FadeIn(ar), run=0.7)
        self.keep(ar)
        self.orbit(0.35, run=1.6)

    # ---- 4. negligible volume ----
    def s_volume(self):
        self.caption(r"Real images occupy a negligible proportion of image space")
        self.say(r"The volume of image space is huge")
        self.go("oblique"); self.floor(-2.8)
        h = 2.4
        edges = _wirecube(h)
        self.act("pop", *[Create(e) for e in edges], run=1.0)
        self.keep(*edges)
        rng = np.random.RandomState(5)
        noise = [Dot3D(rng.uniform(-h, h, 3), radius=0.05, color=K) for _ in range(50)]
        lag = 0.02
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in noise], lag_ratio=lag),
                 run=1.2, n=len(noise), lag=lag, at=0.0)
        self.keep(*noise)
        self.say(r"Images from AI applications fill only a tiny corner")
        tiny = _blk(np.array([1.9, 1.9, 1.9]), (0.3, 0.3, 0.3))
        tiny.set_fill(ACCENT, 0.9)
        self.act("zap", FadeIn(tiny), run=0.7)
        self.keep(tiny)
        self.orbit(0.5, run=2.0)

    # ---- 5. evidence for the manifold hypothesis ----
    def s_evidence(self):
        self.caption(r"The manifold hypothesis: two kinds of evidence")
        self.say(r"The assumption is at least approximately correct")
        self.go("oblique"); self.floor(-2.8)
        h = 2.4
        edges = _wirecube(h)
        self.act("pop", *[Create(e) for e in edges], run=0.9)
        self.keep(*edges)
        b1 = _blk(np.array([-3.4, -2.6, 0.0]), (0.7, 0.7, 0.4)); b1.set_fill(ACCENT, 0.9)
        self.act("pop", FadeIn(b1), run=0.5)
        self.keep(b1)
        self.say(r"First: probability mass is highly concentrated")
        b2 = _blk(np.array([3.4, -2.6, 0.0]), (0.7, 0.7, 0.4)); b2.set_fill(ACCENT, 0.9)
        self.act("pop", FadeIn(b2), run=0.5)
        self.keep(b2)
        ts = np.linspace(-2.0, 2.0, 26)
        rib = [Dot3D(np.array([t, 1.2 * np.sin(1.3 * t), 0.8 * np.cos(1.1 * t)]), radius=0.1, color=ACCENT)
               for t in ts]
        lag = 0.04
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in rib], lag_ratio=lag),
                 run=1.4, n=len(rib), lag=lag, at=0.0)
        self.keep(*rib)
        self.say(r"Realistic inputs lie along a thin manifold")
        self.orbit(0.5, run=2.0)


def main():
    os.chdir(ROOT)
    res = {"preview": "540,540", "reel": "1080,1080", "youtube": "2560,1440", "sq1440": "1440,1440"}.get(os.environ.get("P159_MODE", "reel"), "1080,1080")
    name = "Page159_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page159_timeline.json"
    wav = ROOT / "media/sfx/Page159.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_159.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page159"],
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
