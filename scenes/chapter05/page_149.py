"""Page 149 - 5.9 Stochastic Gradient Descent: cost as a sum, O(m) step, minibatch estimate.
Built on page 145 (same helpers, same sound pipeline).
Run:   python3 scenes/chapter05/page_149.py            (1080x1080 -> assets/user-data/output)
Quick: P149_RES=540,540 python3 scenes/chapter05/page_149.py
"""
import json
import os
import pathlib
import subprocess
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from manim import *
from lib3d import *
from scenes.chapter05.page_143 import write_wav
from scenes.chapter05.page_145 import Page145, build_audio, _forget


class Page149(Page145):
    def construct(self):
        self.setup_stage("oblique", 0.65)
        self._v = "oblique"; self._sub = None
        self.s_sum(); self.wipe()
        self.s_batch(); self.wipe()
        self.s_fixed(); self.wipe()
        self.s_grad(); self.wipe()
        self.wait(0.4)

    def _grid(self, rows, cols, size, text):
        self.go("top")
        grid = matrix_grid(rows, cols, size, ORIGIN)
        run, lag = 1.5, 0.03
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in grid], lag_ratio=lag), run=run,
                 n=rows * cols, lag=lag, at=0.6)
        self.keep(grid)
        cells = [grid_cell(grid, cols, i, j) for i in range(rows) for j in range(cols)]
        return grid, cells

    # ---- cost is a sum over examples, gradient step costs O(m)
    def s_sum(self):
        self.caption("5.9  Stochastic gradient descent")
        self.say(r"Nearly all of deep learning is powered by $\mathrm{SGD}$")
        grid, cells = self._grid(5, 8, 0.6, "")
        self.say(r"The cost is a sum over the $m$ training examples")
        self.wait(0.5)
        self.say(r"$J(\theta)=\frac{1}{m}\sum_{i=1}^{m} L(x^{(i)},y^{(i)},\theta)$")
        self.cue("ping", 0.5, i=0)
        self.play(*[lit(c) for c in cells], run_time=0.8)
        self.wait(0.5)
        self.say(r"One gradient step touches every example: cost $O(m)$")
        self.wait(0.8)
        self.say("With billions of examples, a single step takes far too long")
        self.wait(0.8)

    # ---- minibatch
    def s_batch(self):
        self.caption("The insight: the gradient is an expectation")
        self.say("An expectation can be estimated from a small set of samples")
        grid, cells = self._grid(5, 8, 0.6, "")
        rng = np.random.default_rng(149)
        self.say(r"Sample a minibatch of $m'$ examples, drawn uniformly")
        for k in range(3):
            idx = rng.choice(len(cells), 6, replace=False)
            self.cue("ping", 0.5, i=k)
            self.play(*[lit(cells[i]) for i in idx], run_time=0.5)
            self.wait(0.6)
            self.act("fade", *[unlit(cells[i]) for i in idx], run=0.4)
        self.say("Each step uses a fresh minibatch")
        self.wait(0.6)

    # ---- m' fixed as m grows
    def s_fixed(self):
        self.caption(r"The minibatch size stays fixed")
        self.say(r"$m'$ ranges from one to a few hundred")
        grid, cells = self._grid(8, 12, 0.45, "")
        rng = np.random.default_rng(1492)
        idx = rng.choice(len(cells), 6, replace=False)
        self.cue("ping", 0.5, i=0)
        self.play(*[lit(cells[i]) for i in idx], run_time=0.6)
        self.say(r"$m'$ is held fixed as the training set size $m$ grows")
        self.wait(0.8)
        self.say("Billions of examples can be fit with updates on only a hundred")
        self.wait(0.8)

    # ---- gradient estimate
    def s_grad(self):
        self.caption("The minibatch gives an estimate of the gradient")
        self.say(r"$g=\frac{1}{m'}\nabla_{\theta}\sum_{i=1}^{m'} L(x^{(i)},y^{(i)},\theta)$")
        self.go("oblique")
        self.floor(-0.05)
        f = lambda t: 0.38 * t * t - 1.2
        curve = curve3d(lambda t: np.array([t, f(t), 0.14]), (-3.0, 3.0), K, 5)
        self.act("cut", Create(curve), run=1.2)
        self.keep(curve)
        t0 = 1.6
        p = np.array([t0, f(t0), 0.14])
        pd = sphere(0.16, p)
        self.act("grow", FadeIn(pd), run=0.5)
        self.keep(pd)
        tang = np.array([1.0, 0.76 * t0, 0.0]); tang /= np.linalg.norm(tang)
        down = -tang * 1.8
        true = arrow3d(p + np.array([0, 0, 0.05]), p + np.array([0, 0, 0.05]) + down, color=K)
        self.act("zap", FadeIn(true), run=0.7)
        self.keep(true)
        self.label([(r"$\nabla_{\theta} J$", p + down + np.array([-0.3, 0.6, 0.1]))])
        self.say("Each minibatch gives a slightly different estimate")
        rng = np.random.default_rng(1493)
        ests = []
        for k in range(5):
            noise = np.array([rng.normal(0, 0.28), rng.normal(0, 0.28), 0.0])
            a = arrow3d(p + np.array([0, 0, 0.12]), p + np.array([0, 0, 0.12]) + down + noise, color=ACCENT)
            ests.append(a)
        self.act("stagger", LaggedStart(*[FadeIn(a) for a in ests], lag_ratio=0.4), run=1.6, n=len(ests),
                 lag=0.4, at=0.6)
        self.keep(*ests)
        self.say("The estimates cluster around the true gradient")
        self.cue("ping", 0.5, i=2)
        self.play(Indicate(true, color=ACCENT, scale_factor=1.0), run_time=0.6)
        self.orbit(0.4, run=1.8)
        self.cue("finale", 1.0)
        self.wait(1.0)


# ================================================================ pipeline
def main():
    os.chdir(ROOT)
    res = os.environ.get("P149_RES", "1080,1080")
    name = "Page149_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page149_timeline.json"
    wav = ROOT / "media/sfx/Page149.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_149.log")
    print(f"rendering {res} @30fps ...  (progress: bash lib3d/watch_progress.sh /tmp/page_149.log)")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page149"],
                            env=env, stdout=lf, stderr=subprocess.STDOUT).returncode
    if rc != 0 or not tl.exists():
        print("RENDER FAILED. last lines of", log)
        print("".join(log.read_text().splitlines(True)[-40:]))
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
    print("=== SFX REPORT ===")
    print(f"timeline {data['total']:.2f}s | {len(data['events'])} plays | {len(data['cues'])} cues | {len(slog)} sounds")
    print("sounds:", dict(Counter(n for _, n in slog)))
    print("SAVED:", final)
    print("CHECK SHEET:", sheet)
    subprocess.run(["open", str(final)])
    subprocess.run(["open", str(sheet)])


if __name__ == "__main__":
    main()
