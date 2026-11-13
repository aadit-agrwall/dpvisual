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


def _wirebowl(f, rng=2.6, n=7, m=8):
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


def _grid(c, nx, ny, nz, step=0.5, s=0.36):
    cells = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                p = np.array(c, dtype=float) + np.array([(i - (nx - 1) / 2) * step,
                                                         (j - (ny - 1) / 2) * step,
                                                         (k - (nz - 1) / 2) * step])
                cells.append(_blk(p, (s, s, s)))
    return cells


class Page152(Page146):

    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_noisy(); self.wipe()
        self.s_flat(); self.wipe()
        self.s_taxo(); self.wipe()
        self.s_fail(); self.wipe()
        self.s_curse(); self.wipe()
        self.wait(0.4)

    # ---- 1. cost cannot be evaluated: approximate gradients ----
    def s_noisy(self):
        self.caption(r"A cost we cannot evaluate: approximate its gradients")
        self.say(r"We can still minimize it with iterative numerical optimization")
        self.go("oblique"); self.floor(-0.2)
        f = lambda u, v: 0.15 * (u * u + v * v)
        wire = _wirebowl(f)
        run, lag = 1.4, 0.004
        self.act("stagger", LaggedStart(*[FadeIn(w) for w in wire], lag_ratio=lag),
                 run=run, n=len(wire), lag=lag, at=0.0)
        self.keep(*wire)
        rng = np.random.RandomState(3)
        p = np.array([-2.2, 1.8])
        path = [p.copy()]
        for _ in range(6):
            g = np.array([0.3 * p[0], 0.3 * p[1]]) + rng.normal(0, 0.15, 2)
            p = p - 1.4 * g
            path.append(p.copy())
        z = lambda q: f(q[0], q[1]) + 0.14
        ball = Dot3D(np.array([path[0][0], path[0][1], z(path[0])]), radius=0.14, color=ACCENT)
        self.act("pop", FadeIn(ball), run=0.5)
        self.keep(ball)
        self.say(r"As long as we can approximate the gradients")
        for a, b in zip(path[:-1], path[1:]):
            tr = segment(np.array([a[0], a[1], z(a)]), np.array([b[0], b[1], z(b)]), width=4)
            self.act("cut", ball.animate.move_to(np.array([b[0], b[1], z(b)])), FadeIn(tr), run=0.45)
            self.keep(tr)
        self.orbit(0.35, run=1.6)

    # ---- 2. flat regions need special-case optimizers ----
    def s_flat(self):
        self.caption(r"Decision trees and $k$-means: flat cost regions")
        self.say(r"Gradient-based optimizers cannot move on a flat region")
        self.go("oblique"); self.floor(-0.5)
        xs = [-3.0, -1.0, 1.0, 3.0]
        hs = [1.2, 0.8, 0.4, 0.0]
        steps = [_blk((x, 0.0, h), (1.6, 1.6, 0.2)) for x, h in zip(xs, hs)]
        self.act("pop", *[FadeIn(s) for s in steps], run=1.0)
        self.keep(*steps)
        top = lambda i: np.array([xs[i], 0.0, hs[i] + 0.26])
        ball = Dot3D(top(0), radius=0.14, color=ACCENT)
        self.act("pop", FadeIn(ball), run=0.5)
        self.keep(ball)
        self.act("zap", ball.animate.shift(np.array([0.0, 0.0, 0.0])), run=0.5)
        self.say(r"A special-case optimizer jumps between regions instead")
        for i in (1, 2, 3):
            self.act("cut", ball.animate.move_to(top(i)), run=0.55)
        self.act("zap", steps[3].animate.set_fill(ACCENT, 0.9), run=0.6)
        self.orbit(0.35, run=1.6)

    # ---- 3. a taxonomy instead of a long list ----
    def s_taxo(self):
        self.caption(r"One recipe: a taxonomy of methods")
        self.say(r"Not a long list of algorithms with separate justifications")
        self.go("oblique"); self.floor(-0.6)
        row = [(-3.0 + i * 1.0, -2.0, 0.0) for i in range(7)]
        tree = [(0.0, 1.8, 0.0), (-2.0, 0.3, 0.0), (2.0, 0.3, 0.0),
                (-3.0, -1.2, 0.0), (-1.0, -1.2, 0.0), (1.0, -1.2, 0.0), (3.0, -1.2, 0.0)]
        blocks = [_blk(p, (0.6, 0.6, 0.4)) for p in row]
        self.act("pop", *[FadeIn(b) for b in blocks], run=1.0)
        self.keep(*blocks)
        self.say(r"Related tasks that work for similar reasons")
        self.act("cut", *[b.animate.move_to(np.array(t, dtype=float)) for b, t in zip(blocks, tree)], run=1.4)
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        segs = [segment(np.array(tree[a], dtype=float), np.array(tree[b], dtype=float), width=3) for a, b in edges]
        self.act("pop", *[Create(sg) for sg in segs], run=0.9)
        self.keep(*segs)
        self.act("zap", blocks[0].animate.set_fill(ACCENT, 0.9), run=0.6)
        self.orbit(0.35, run=1.6)

    # ---- 4. simple algorithms fall short on AI tasks ----
    def s_fail(self):
        self.caption(r"5.11 Challenges motivating deep learning")
        self.say(r"Simple algorithms work well on many important problems")
        self.go("oblique"); self.floor(-0.8)
        small = [_blk((-3.4, y, 0.0), (0.6, 0.6, 0.5)) for y in (-1.0, 0.0, 1.0)]
        self.act("pop", *[FadeIn(b) for b in small], run=0.8)
        self.keep(*small)
        self.act("zap", *[b.animate.set_fill(ACCENT, 0.9) for b in small], run=0.6)
        self.say(r"But they have not solved recognizing speech or objects")
        xs = np.linspace(-1.2, 3.0, 28)
        wave = [segment(np.array([a, 1.4, 0.8 * np.sin(2.6 * a)]), np.array([b, 1.4, 0.8 * np.sin(2.6 * b)]), width=3)
                for a, b in zip(xs[:-1], xs[1:])]
        obj = _blk((1.0, -1.2, 0.0), (1.6, 1.2, 1.0))
        self.act("cut", *[Create(w) for w in wave], FadeIn(obj), run=1.3)
        self.keep(*wave, obj)
        arr = arrow3d(np.array([-2.8, 0.0, 0.1]), np.array([-1.6, 0.0, 0.1]), color=ACCENT)
        self.act("zap", FadeIn(arr), run=0.7)
        self.keep(arr)
        self.say(r"Traditional algorithms fail to generalize on such AI tasks")
        self.orbit(0.35, run=1.6)

    # ---- 5. curse of dimensionality ----
    def s_curse(self):
        self.caption(r"The curse of dimensionality")
        self.say(r"The number of possible configurations grows exponentially")
        self.go("oblique"); self.floor(-1.2)
        g1 = _grid((-4.0, 0.0, 0.0), 4, 1, 1)
        g2 = _grid((-1.2, 0.0, 0.0), 4, 4, 1)
        g3 = _grid((2.6, 0.0, 0.0), 4, 4, 4)
        for g, t in ((g1, 0.8), (g2, 1.0), (g3, 1.2)):
            lag = 0.02
            self.act("stagger", LaggedStart(*[FadeIn(c) for c in g], lag_ratio=lag),
                     run=t, n=len(g), lag=lag, at=0.0)
            self.keep(*g)
        self.say(r"More variables, exponentially more distinct configurations")
        self.act("zap", g1[-1].animate.set_fill(ACCENT, 0.9), g2[-1].animate.set_fill(ACCENT, 0.9),
                 g3[-1].animate.set_fill(ACCENT, 0.9), run=0.7)
        self.orbit(0.35, run=1.8)


def main():
    os.chdir(ROOT)
    res = {"preview": "540,540", "reel": "1080,1080", "youtube": "2560,1440", "sq1440": "1440,1440"}.get(os.environ.get("P152_MODE", "reel"), "1080,1080")
    name = "Page152_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page152_timeline.json"
    wav = ROOT / "media/sfx/Page152.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_152.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page152"],
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
