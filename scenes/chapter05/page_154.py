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


class Page154(Page146):

    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_nearest(); self.wipe()
        self.s_prior(); self.wipe()
        self.s_cond(); self.wipe()
        self.s_avg(); self.wipe()
        self.s_knn(); self.wipe()
        self.wait(0.4)

    # ---- 1. output at the nearest training point ----
    def s_nearest(self):
        self.caption(r"Output at a new point is about the output at the nearest training point")
        self.say(r"Many traditional algorithms simply assume this")
        self.go("oblique"); self.floor(-0.1)
        rng = np.random.RandomState(8)
        P = [np.array([rng.uniform(-3, 3), rng.uniform(-2.2, 2.2), 0.1]) for _ in range(10)]
        dots = [Dot3D(p, radius=0.1, color=K) for p in P]
        lag = 0.08
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag),
                 run=1.2, n=len(dots), lag=lag, at=0.0)
        self.keep(*dots)
        q = np.array([0.3, 0.2, 0.1])
        new = Dot3D(q, radius=0.15, color=ACCENT)
        self.act("pop", FadeIn(new), run=0.5)
        self.keep(new)
        self.say(r"Find the nearest training point")
        j = int(np.argmin([np.linalg.norm(p - q) for p in P]))
        sg = segment(q, P[j], width=5)
        self.act("cut", Create(sg), run=0.8)
        self.keep(sg)
        self.act("zap", dots[j].animate.set_color(ACCENT), run=0.6)
        self.orbit(0.35, run=1.6)

    # ---- 2. smoothness prior ----
    def s_prior(self):
        self.caption(r"Smoothness prior, or local constancy prior")
        self.say(r"The function we learn should not change much within a small region")
        self.go("oblique"); self.floor(-0.8)
        f = lambda u, v: 0.25 * np.sin(0.8 * u) + 0.2 * np.cos(0.8 * v)
        wire = _wire(f)
        lag = 0.004
        self.act("stagger", LaggedStart(*[FadeIn(w) for w in wire], lag_ratio=lag),
                 run=1.4, n=len(wire), lag=lag, at=0.0)
        self.keep(*wire)
        self.say(r"Within a small neighborhood the value stays nearly the same")
        patch = _blk((0.6, 0.4, f(0.6, 0.4)), (0.9, 0.9, 0.08))
        self.act("pop", FadeIn(patch), run=0.5)
        self.keep(patch)
        self.act("zap", patch.animate.set_fill(ACCENT, 0.9), run=0.6)
        self.orbit(0.35, run=1.6)

    # ---- 3. eq 5.103 ----
    def s_cond(self):
        self.caption(r"Condition: $f^{*}(\mathbf{x})\approx f^{*}(\mathbf{x}+\epsilon)$")
        self.say(r"For most configurations $\mathbf{x}$ and a small change $\epsilon$")
        self.go("oblique"); self.floor(-1.0)
        g = lambda x: 0.7 * np.sin(0.9 * x)
        xs = np.linspace(-3.2, 3.2, 40)
        curve = [segment(np.array([a, 0.0, g(a)]), np.array([b, 0.0, g(b)]), width=4)
                 for a, b in zip(xs[:-1], xs[1:])]
        lag = 0.02
        self.act("stagger", LaggedStart(*[Create(c) for c in curve], lag_ratio=lag),
                 run=1.4, n=len(curve), lag=lag, at=0.0)
        self.keep(*curve)
        x0, x1 = 0.5, 0.95
        a = Dot3D(np.array([x0, 0.0, g(x0)]), radius=0.14, color=ACCENT)
        b = Dot3D(np.array([x1, 0.0, g(x1)]), radius=0.14, color=ACCENT)
        self.act("pop", FadeIn(a), FadeIn(b), run=0.6)
        self.keep(a, b)
        self.eq(r"f^{*}(\mathbf{x})\approx f^{*}(\mathbf{x}+\epsilon)")
        self.say(r"A good answer at $\mathbf{x}$ is probably good in its neighborhood")
        eps = segment(np.array([x0, 0.0, -0.9]), np.array([x1, 0.0, -0.9]), width=6)
        self.act("cut", Create(eps), run=0.6)
        self.keep(eps)
        self.orbit(0.35, run=1.6)

    # ---- 4. average several good answers ----
    def s_avg(self):
        self.caption(r"Several good answers in a neighborhood")
        self.say(r"Combine them by averaging or interpolation")
        self.go("oblique"); self.floor(-0.2)
        rng = np.random.RandomState(2)
        ang = np.linspace(0, 2 * np.pi, 6)[:-1]
        P = [np.array([1.8 * np.cos(t), 1.8 * np.sin(t), rng.uniform(0.4, 1.4)]) for t in ang]
        dots = [Dot3D(p, radius=0.12, color=K) for p in P]
        self.act("pop", *[FadeIn(d) for d in dots], run=0.8)
        self.keep(*dots)
        mean = float(np.mean([p[2] for p in P]))
        new = Dot3D(np.array([0.0, 0.0, 0.0]), radius=0.16, color=ACCENT)
        self.act("pop", FadeIn(new), run=0.5)
        self.keep(new)
        lines = [segment(p, np.array([0.0, 0.0, 0.0]), width=2) for p in P]
        self.act("cut", *[Create(l) for l in lines], run=0.9)
        self.keep(*lines)
        self.say(r"The answer should agree with as many of them as possible")
        self.act("zap", new.animate.move_to(np.array([0.0, 0.0, mean])), run=1.0)
        self.orbit(0.35, run=1.6)

    # ---- 5. k-nearest neighbors ----
    def s_knn(self):
        self.caption(r"$k$-nearest neighbors: an extreme case of local constancy")
        self.say(r"Constant over each region with the same set of nearest neighbors")
        self.go("oblique"); self.floor(-0.1)
        n, step = 9, 0.7
        seeds = [np.array([-1.6, -1.2]), np.array([1.4, -0.6]), np.array([0.0, 1.6])]
        hts = [0.3, 0.9, 0.6]
        cells, owner = [], []
        for i in range(n):
            for j in range(n):
                p = np.array([(i - 4) * step, (j - 4) * step])
                k = int(np.argmin([np.linalg.norm(p - s) for s in seeds]))
                h = hts[k]
                cells.append(_blk((p[0], p[1], h / 2), (0.62, 0.62, h)))
                owner.append(k)
        lag = 0.01
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in cells], lag_ratio=lag),
                 run=1.6, n=len(cells), lag=lag, at=0.0)
        self.keep(*cells)
        self.say(r"The predictor is flat inside each region")
        reg = [c for c, k in zip(cells, owner) if k == 1]
        self.act("zap", *[c.animate.set_fill(ACCENT, 0.9) for c in reg], run=0.8)
        self.orbit(0.5, run=2.0)


def main():
    os.chdir(ROOT)
    res = {"preview": "540,540", "reel": "1080,1080", "youtube": "2560,1440", "sq1440": "1440,1440"}.get(os.environ.get("P154_MODE", "reel"), "1080,1080")
    name = "Page154_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page154_timeline.json"
    wav = ROOT / "media/sfx/Page154.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_154.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page154"],
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
