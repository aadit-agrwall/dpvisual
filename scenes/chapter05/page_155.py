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


class Page155(Page146):

    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_k1(); self.wipe()
        self.s_kernel(); self.wipe()
        self.s_tree(); self.wipe()
        self.s_ok(); self.wipe()
        self.s_checker(); self.wipe()
        self.wait(0.4)

    # ---- 1. k = 1: regions <= examples ----
    def s_k1(self):
        self.caption(r"Nearest neighbor with $k=1$")
        self.say(r"The number of distinguishable regions cannot exceed the number of examples")
        self.go("oblique"); self.floor(-0.1)
        seeds = [np.array(p) for p in [(-2.0, -1.4), (0.0, -1.8), (2.0, -1.0),
                                       (-1.6, 1.4), (0.4, 1.0), (2.2, 1.8)]]
        n, step = 9, 0.7
        cells, owner = [], []
        for i in range(n):
            for j in range(n):
                p = np.array([(i - 4) * step, (j - 4) * step])
                k = int(np.argmin([np.linalg.norm(p - s) for s in seeds]))
                cells.append(_blk((p[0], p[1], 0.05), (0.62, 0.62, 0.1)))
                owner.append(k)
        lag = 0.01
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.4, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        dots = [Dot3D(np.array([s[0], s[1], 0.25]), radius=0.14, color=K) for s in seeds]
        self.act("pop", *[FadeIn(d) for d in dots], run=0.7)
        self.keep(*dots)
        self.say(r"Each training example defines at most one region")
        for k in range(len(seeds)):
            reg = [c for c, o in zip(cells, owner) if o == k]
            self.act("zap", *[c.animate.set_fill(ACCENT, 0.8) for c in reg], run=0.35)
            self.act("zap", *[c.animate.set_fill("#FFFFFF", 1) for c in reg], run=0.2)
        self.act("zap", *[c.animate.set_fill(ACCENT, 0.5) for c in cells], run=0.5)
        self.orbit(0.35, run=1.6)

    # ---- 2. local kernels ----
    def s_kernel(self):
        self.caption(r"Local kernels: a similarity function")
        self.say(r"$k(\mathbf{u},\mathbf{v})$ is large when $\mathbf{u}=\mathbf{v}$ and decays with distance")
        self.go("oblique"); self.floor(-0.2)
        f = lambda u, v: 1.6 * np.exp(-0.5 * (u * u + v * v))
        wire = _wire(f, rng=2.6, n=9, m=10)
        lag = 0.004
        self.act("stagger", LaggedStart(*[FadeIn(w) for w in wire], lag_ratio=lag),
                 run=1.4, n=len(wire), lag=lag, at=0.0)
        self.keep(*wire)
        P = [np.array(p) for p in [(-2.0, 1.0), (1.6, -1.4), (2.0, 1.6), (-1.2, -1.8)]]
        dots = [Dot3D(np.array([p[0], p[1], 0.1]), radius=0.1, color=K) for p in P]
        self.act("pop", *[FadeIn(d) for d in dots], run=0.7)
        self.keep(*dots)
        self.say(r"Template matching: how closely a test example resembles each training example")
        test = Dot3D(np.array([0.0, 0.0, f(0, 0) + 0.1]), radius=0.16, color=ACCENT)
        self.act("pop", FadeIn(test), run=0.5)
        self.keep(test)
        lines = [segment(np.array([0.0, 0.0, f(0, 0) + 0.1]), np.array([p[0], p[1], 0.1]),
                         width=max(1, int(6 * np.exp(-0.5 * (p[0] ** 2 + p[1] ** 2)) + 1))) for p in P]
        self.act("cut", *[Create(l) for l in lines], run=0.9)
        self.keep(*lines)
        self.orbit(0.35, run=1.6)

    # ---- 3. decision trees ----
    def s_tree(self):
        self.caption(r"Decision trees: as many regions as leaves")
        self.say(r"A tree with at least $n$ leaves needs at least $n$ training examples")
        self.go("oblique"); self.floor(-1.0)
        nodes = [(0.0, 0.0, 2.4),
                 (-2.0, 0.0, 1.2), (2.0, 0.0, 1.2),
                 (-3.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (3.0, 0.0, 0.0)]
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        blocks = [_blk(p, (0.7, 0.7, 0.5)) for p in nodes]
        self.act("pop", *[FadeIn(b) for b in blocks], run=1.0)
        self.keep(*blocks)
        segs = [segment(np.array(nodes[a], dtype=float), np.array(nodes[b], dtype=float), width=3)
                for a, b in edges]
        self.act("cut", *[Create(s) for s in segs], run=0.9)
        self.keep(*segs)
        self.say(r"A separate parameter in each region")
        leaves = blocks[3:]
        self.act("zap", *[b.animate.set_fill(ACCENT, 0.9) for b in leaves], run=0.7)
        dots = [Dot3D(np.array(p, dtype=float) + np.array([0.0, 0.0, -0.8]), radius=0.13, color=K)
                for p in nodes[3:]]
        self.act("pop", *[FadeIn(d) for d in dots], run=0.7)
        self.keep(*dots)
        self.say(r"A multiple of $n$ is needed for statistical confidence")
        self.orbit(0.35, run=1.6)

    # ---- 4. O(k) regions need O(k) examples ----
    def s_ok(self):
        self.caption(r"$O(k)$ regions need $O(k)$ examples")
        self.say(r"Typically $O(k)$ parameters, with $O(1)$ parameters for each region")
        self.go("oblique"); self.floor(-0.8)
        k = 8
        xs = [(-3.5 + i * 1.0) for i in range(k)]
        regions = [_blk((x, 0.0, 0.05), (0.85, 1.6, 0.1)) for x in xs]
        lag = 0.08
        self.act("stagger", LaggedStart(*[FadeIn(r) for r in regions], lag_ratio=lag),
                 run=1.2, n=len(regions), lag=lag, at=0.0)
        self.keep(*regions)
        params = [_blk((x, 0.0, 0.6), (0.35, 0.35, 0.35)) for x in xs]
        self.act("pop", *[FadeIn(p) for p in params], run=0.8)
        self.keep(*params)
        dots = [Dot3D(np.array([x, -0.5, 0.2]), radius=0.1, color=K) for x in xs]
        self.act("pop", *[FadeIn(d) for d in dots], run=0.8)
        self.keep(*dots)
        self.eq(r"O(k)\ \mathrm{regions}\Rightarrow O(k)\ \mathrm{examples}")
        self.say(r"Each training example defines at most one region")
        self.act("zap", *[p.animate.set_fill(ACCENT, 0.9) for p in params], run=0.7)
        self.orbit(0.35, run=1.6)

    # ---- 5. checkerboard ----
    def s_checker(self):
        self.caption(r"A checkerboard: many variations, a simple structure")
        self.say(r"Smoothness alone cannot extend the pattern to unseen squares")
        self.go("oblique"); self.floor(-0.1)
        n, step = 8, 0.62
        cells, cen = [], []
        for i in range(n):
            for j in range(n):
                p = np.array([(i - 3.5) * step, (j - 3.5) * step])
                c = _blk((p[0], p[1], 0.05), (0.58, 0.58, 0.1))
                if (i + j) % 2 == 0:
                    c.set_fill("#222222", 1)
                cells.append(c); cen.append(p)
        lag = 0.01
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.5, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        self.say(r"A few training examples cover only a few squares")
        occ = [9, 20, 27, 38, 50]
        dots = [Dot3D(np.array([cen[i][0], cen[i][1], 0.25]), radius=0.1, color=ACCENT) for i in occ]
        self.act("pop", *[FadeIn(d) for d in dots], run=0.7)
        self.keep(*dots)
        self.act("zap", *[cells[i].animate.set_stroke(ACCENT, 5) for i in occ], run=0.6)
        self.say(r"Squares without examples: no guarantee of the right color")
        new = Dot3D(np.array([cen[45][0], cen[45][1], 0.4]), radius=0.15, color=K)
        self.act("pop", FadeIn(new), run=0.5)
        self.keep(new)
        self.act("cut", new.animate.shift(np.array([0.0, 0.0, -0.1])), run=0.4)
        self.orbit(0.5, run=2.0)


def main():
    os.chdir(ROOT)
    res = {"preview": "540,540", "reel": "1080,1080", "youtube": "2560,1440", "sq1440": "1440,1440"}.get(os.environ.get("P155_MODE", "reel"), "1080,1080")
    name = "Page155_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page155_timeline.json"
    wav = ROOT / "media/sfx/Page155.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_155.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page155"],
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
