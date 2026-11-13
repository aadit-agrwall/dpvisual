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


ACC = "#1F4FFF"

def _on(m):
    m.set_fill(ACC, 0.9)
    m.set_stroke("#000000", 2)
    return m

def _off(m):
    m.set_fill("#FFFFFF", 1)
    m.set_stroke("#000000", 2)
    return m

def _blk(pos, kind="box"):
    if kind == "ball":
        m = Sphere(radius=0.55, resolution=(12, 12))
    elif kind == "cyl":
        m = Cylinder(radius=0.55, height=1.0, resolution=(12, 12))
    else:
        m = Prism(dimensions=[1.5, 1.0, 0.6])
    _off(m)
    m.move_to(pos)
    return m

def _dot(p, r=0.09):
    d = Dot3D(point=np.array(p, dtype=float), radius=r)
    d.set_color("#000000")
    return d

def _bowl(f, rng=2.6):
    s = Surface(lambda u, v: np.array([u, v, f(u, v)]),
                u_range=[-rng, rng], v_range=[-rng, rng], resolution=(8, 8))
    s.set_fill("#FFFFFF", 0.55)
    s.set_stroke("#000000", 1)
    return s


class Page151(Page146):

    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_recipe(); self.wipe()
        self.s_linreg(); self.wipe()
        self.s_swap(); self.wipe()
        self.s_decay(); self.wipe()
        self.s_nonlin(); self.wipe()
        self.s_pca1(); self.wipe()
        self.wait(0.4)

    # ---- 1. the recipe ----
    def s_recipe(self):
        self.caption(r"Building a machine learning algorithm: a simple recipe")
        self.say(r"Combine a dataset, a cost function, an optimizer and a model")
        self.go("oblique"); self.floor(-0.6)
        pos = [(-3.0, -1.2, 0), (-1.0, 0.4, 0), (1.0, -0.4, 0), (3.0, 1.2, 0)]
        blocks = [_blk(p) for p in pos]
        self.act("pop", *[FadeIn(b, shift=DOWN * 0.6) for b in blocks], run=1.2)
        self.keep(*blocks)
        for b, t in zip(blocks, [r"1. A specification of a dataset", r"2. A cost function",
                                 r"3. An optimization procedure", r"4. A model"]):
            self.say(t)
            self.act("zap", b.animate.set_fill(ACCENT, 0.9), run=0.6)
        self.orbit(0.35, run=1.6)

    # ---- 2. linear regression ----
    def s_linreg(self):
        self.caption(r"Example: linear regression")
        self.say(r"A dataset consisting of $\mathbf{X}$ and $\mathbf{y}$")
        self.go("oblique"); self.floor(-1.6)
        rng = np.random.RandomState(5)
        pts = [_dot([x, y, 0.5 * x + rng.normal(0, 0.25)])
               for x, y in zip(rng.uniform(-2.8, 2.8, 16), rng.uniform(-2.0, 2.0, 16))]
        run, lag = 1.4, 0.05
        self.act("stagger", LaggedStart(*[FadeIn(p) for p in pts], lag_ratio=lag),
                 run=run, n=len(pts), lag=lag, at=0.0)
        self.keep(*pts)
        self.eq(r"J(\mathbf{w},b)=-\mathbb{E}_{\mathbf{x},y\sim\hat{p}_{\mathrm{data}}}\log p_{\mathrm{model}}(y\mid\mathbf{x})")
        self.say(r"Model: $p_{\mathrm{model}}(y\mid\mathbf{x})=\mathcal{N}(y;\mathbf{x}^{\top}\mathbf{w}+b,1)$")
        plane = _bowl(lambda u, v: 0.5 * u, 2.8)
        self.act("cut", Create(plane), run=1.2)
        self.keep(plane)
        self.say(r"Optimizer: solve for zero gradient with the normal equations")
        self.act("zap", plane.animate.set_fill(ACCENT, 0.45), run=0.6)
        self.orbit(0.35, run=1.6)

    # ---- 3. replace components ----
    def s_swap(self):
        self.caption(r"Replace any component, mostly independently")
        self.say(r"Swap the model for a different one")
        self.go("oblique"); self.floor(-0.6)
        pos = [(-3.0, -1.2, 0), (-1.0, 0.4, 0), (1.0, -0.4, 0), (3.0, 1.2, 0)]
        blocks = [_blk(p) for p in pos]
        self.act("pop", *[FadeIn(b) for b in blocks], run=1.0)
        self.keep(*blocks)
        new = _on(_blk(pos[3], "ball"))
        self.act("zap", ReplacementTransform(blocks[3], new), run=0.9)
        _forget(self, [blocks[3]]); self.keep(new)
        self.say(r"Swap the cost function")
        new2 = _on(_blk(pos[1], "cyl"))
        self.act("zap", ReplacementTransform(blocks[1], new2), run=0.9)
        _forget(self, [blocks[1]]); self.keep(new2)
        self.say(r"Doing this gives a wide range of algorithms")
        self.orbit(0.35, run=1.6)

    # ---- 4. weight decay ----
    def s_decay(self):
        self.caption(r"Cost: negative log-likelihood plus regularization")
        self.say(r"Minimizing the negative log-likelihood gives maximum likelihood")
        self.go("oblique"); self.floor(-0.3)
        a, lam, cx, cy = 0.15, 0.12, 1.2, 0.8
        bowl = _bowl(lambda u, v: a * ((u - cx) ** 2 + (v - cy) ** 2))
        self.act("pop", Create(bowl), run=1.2)
        self.keep(bowl)
        ball = _dot([cx, cy, 0.0], 0.14); ball.set_color(ACCENT)
        self.act("pop", FadeIn(ball), run=0.5)
        self.keep(ball)
        self.eq(r"J(\mathbf{w},b)=\lambda\|\mathbf{w}\|_2^2-\mathbb{E}_{\mathbf{x},y\sim\hat{p}_{\mathrm{data}}}\log p_{\mathrm{model}}(y\mid\mathbf{x})")
        self.say(r"Weight decay adds a term to the cost")
        bowl2 = _bowl(lambda u, v: a * ((u - cx) ** 2 + (v - cy) ** 2) + lam * (u ** 2 + v ** 2))
        mx, my = a * cx / (a + lam), a * cy / (a + lam)
        self.act("zap", Transform(bowl, bowl2), ball.animate.move_to(np.array([mx, my, 0.0])), run=1.4)
        self.say(r"This still allows closed form optimization")
        self.orbit(0.35, run=1.6)

    # ---- 5. nonlinear model ----
    def s_nonlin(self):
        self.caption(r"Nonlinear model: no closed form")
        self.say(r"Use an iterative numerical optimizer such as gradient descent")
        self.go("oblique"); self.floor(-0.5)
        f = lambda u, v: 0.35 * np.sin(1.2 * u) * np.cos(1.2 * v) + 0.08 * (u ** 2 + v ** 2)
        surf = _bowl(f)
        self.act("pop", Create(surf), run=1.2)
        self.keep(surf)
        p = np.array([-2.2, 1.8]); steps = [p.copy()]
        for _ in range(7):
            e = 1e-3
            g = np.array([(f(p[0] + e, p[1]) - f(p[0] - e, p[1])) / (2 * e),
                          (f(p[0], p[1] + e) - f(p[0], p[1] - e)) / (2 * e)])
            p = p - 1.6 * g
            steps.append(p.copy())
        ball = _dot([steps[0][0], steps[0][1], f(*steps[0]) + 0.12], 0.14); ball.set_color(ACCENT)
        self.act("pop", FadeIn(ball), run=0.5)
        self.keep(ball)
        for s in steps[1:]:
            self.act("cut", ball.animate.move_to(np.array([s[0], s[1], f(*s) + 0.12])), run=0.45)
        self.orbit(0.35, run=1.6)

    # ---- 6. first PCA vector ----
    def s_pca1(self):
        self.caption(r"Unsupervised learning: a dataset with only $\mathbf{X}$")
        self.say(r"We can obtain the first PCA vector with a loss function")
        self.go("oblique"); self.floor(-1.2)
        rng = np.random.RandomState(11)
        w = np.array([0.8, 0.5, 0.3]); w = w / np.linalg.norm(w)
        pts = [rng.uniform(-3, 3) * w + rng.normal(0, 0.45, 3) for _ in range(14)]
        dots = [_dot(q) for q in pts]
        run, lag = 1.3, 0.06
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag),
                 run=run, n=len(dots), lag=lag, at=0.0)
        self.keep(*dots)
        self.eq(r"J(\mathbf{w})=\mathbb{E}_{\mathbf{x}\sim\hat{p}_{\mathrm{data}}}\|\mathbf{x}-r(\mathbf{x};\mathbf{w})\|_2^2")
        self.say(r"The model has $\mathbf{w}$ with norm one")
        arr = arrow3d(-3.4 * w, 3.4 * w, color=ACCENT)
        self.act("cut", FadeIn(arr), run=0.9)
        self.keep(arr)
        segs = [segment(q, float(np.dot(q, w)) * w, width=2) for q in pts]
        self.act("pop", *[Create(sg) for sg in segs], run=1.2)
        self.keep(*segs)
        self.eq(r"r(\mathbf{x})=\mathbf{w}^{\top}\mathbf{x}\mathbf{w}")
        self.say(r"Reconstruction function")
        self.act("zap", *[d.animate.set_color(ACCENT) for d in dots], run=0.6)
        self.orbit(0.35, run=1.6)


def main():
    os.chdir(ROOT)
    res = {"preview": "540,540", "reel": "1080,1080", "youtube": "2560,1440", "sq1440": "1440,1440"}.get(os.environ.get("P151_MODE", "reel"), "1080,1080")
    name = "Page151_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page151_timeline.json"
    wav = ROOT / "media/sfx/Page151.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_151.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page151"],
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
