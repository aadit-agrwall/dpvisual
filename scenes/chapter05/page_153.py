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


def _cells(nx, ny, nz, step, s, h=None, c=(0.0, 0.0, 0.0)):
    cells, cen = [], []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p = np.array(c, dtype=float) + np.array([(i - (nx - 1) / 2) * step,
                                                         (j - (ny - 1) / 2) * step,
                                                         (k - (nz - 1) / 2) * step])
                cells.append(_blk(p, (s, s, h if h else s)))
                cen.append(p)
    return cells, cen


class Page153(Page146):

    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_one(); self.wipe()
        self.s_two(); self.wipe()
        self.s_three(); self.wipe()
        self.s_count(); self.wipe()
        self.s_empty(); self.wipe()
        self.wait(0.4)

    # ---- 1. Figure 5.9 left: one dimension ----
    def s_one(self):
        self.caption(r"Figure 5.9: the number of configurations grows exponentially")
        self.say(r"One dimension: we only distinguish 10 regions of interest")
        self.go("oblique"); self.floor(-0.3)
        cells, _ = _cells(10, 1, 1, 0.95, 0.8, h=0.2)
        lag = 0.08
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.4, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        self.say(r"With enough examples in each region, we generalize easily")
        self.act("zap", *[c.animate.set_fill(ACCENT, 0.9) for c in cells], run=0.8)
        self.orbit(0.35, run=1.6)

    # ---- 2. Figure 5.9 center: two dimensions ----
    def s_two(self):
        self.caption(r"Two dimensions: $10\times 10=100$ regions")
        self.say(r"It is harder to distinguish 10 values of each variable")
        self.go("oblique"); self.floor(-0.3)
        cells, _ = _cells(10, 10, 1, 0.5, 0.4, h=0.15)
        lag = 0.01
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.6, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        self.say(r"We need at least as many examples as regions")
        diag = [cells[i * 10 + i] for i in range(10)]
        self.act("zap", *[c.animate.set_fill(ACCENT, 0.9) for c in diag], run=0.8)
        self.orbit(0.35, run=1.6)

    # ---- 3. Figure 5.9 right: three dimensions ----
    def s_three(self):
        self.caption(r"Three dimensions: $10^3=1000$ regions")
        self.say(r"The cube grows to 1000 regions and at least as many examples")
        self.go("oblique"); self.floor(-2.6)
        h = 2.4
        corners = [np.array([a, b, c]) * h for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]
        edges = [segment(p, q, width=3) for i, p in enumerate(corners) for q in corners[i + 1:]
                 if np.sum(np.abs(p - q) > 1e-6) == 1]
        self.act("pop", *[Create(e) for e in edges], run=1.0)
        self.keep(*edges)
        rng = np.random.RandomState(7)
        cubes = [_blk(rng.uniform(-2.0, 2.0, 3), (0.35, 0.35, 0.35)) for _ in range(60)]
        lag = 0.02
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cubes], lag_ratio=lag),
                 run=1.5, n=len(cubes), lag=lag, at=0.0)
        self.keep(*cubes)
        self.eq(r"O(v^{d})")
        self.say(r"For $d$ dimensions and $v$ values per axis we need $O(v^d)$ regions")
        self.act("zap", *[c.animate.set_fill(ACCENT, 0.9) for c in cubes[:10]], run=0.7)
        self.orbit(0.5, run=2.0)

    # ---- 4. count examples in the same cell ----
    def s_count(self):
        self.caption(r"A statistical challenge")
        self.say(r"Organize the input space into a grid of cells")
        self.go("oblique"); self.floor(-0.3)
        cells, cen = _cells(6, 6, 1, 0.8, 0.7, h=0.1)
        lag = 0.02
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.2, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        rng = np.random.RandomState(4)
        occ = [0, 1, 6, 7, 8, 13, 14]
        dots = []
        for ci in occ:
            for _ in range(2):
                p = cen[ci] + np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), 0.15])
                dots.append(Dot3D(p, radius=0.07, color=K))
        self.act("pop", *[FadeIn(d) for d in dots], run=0.8)
        self.keep(*dots)
        self.say(r"A new input: inspect the training examples in its cell")
        new = Dot3D(cen[7] + np.array([0.0, 0.0, 0.5]), radius=0.13, color=ACCENT)
        self.act("pop", FadeIn(new), run=0.5)
        self.keep(new)
        self.act("zap", cells[7].animate.set_fill(ACCENT, 0.6), run=0.6)
        self.eq(r"\hat{p}(\mathbf{x})=\frac{\mathrm{count\ in\ cell}}{N}")
        self.say(r"Density: examples in the same cell divided by all examples")
        self.say(r"Classification: the most common class. Regression: the average")
        self.orbit(0.35, run=1.6)

    # ---- 5. most cells are empty ----
    def s_empty(self):
        self.caption(r"Cells with no training example")
        self.say(r"In high dimensions, there are far more configurations than examples")
        self.go("oblique"); self.floor(-1.8)
        cells, cen = _cells(4, 4, 4, 0.85, 0.5)
        lag = 0.015
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.4, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        occ = [0, 21, 26, 38, 43, 60]
        self.act("zap", *[cells[i].animate.set_fill(ACCENT, 0.9) for i in occ], run=0.7)
        self.say(r"A typical cell has no training example")
        new = Dot3D(cen[30] + np.array([0.0, 0.0, 0.55]), radius=0.16, color=K)
        self.act("pop", FadeIn(new), run=0.5)
        self.keep(new)
        self.say(r"What can we say about these new configurations?")
        self.act("cut", new.animate.shift(np.array([0.0, 0.0, -0.15])), run=0.5)
        self.orbit(0.5, run=2.0)


def main():
    os.chdir(ROOT)
    res = {"preview": "540,540", "reel": "1080,1080", "youtube": "2560,1440", "sq1440": "1440,1440"}.get(os.environ.get("P153_MODE", "reel"), "1080,1080")
    name = "Page153_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page153_timeline.json"
    wav = ROOT / "media/sfx/Page153.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_153.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page153"],
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
