"""Page 150 - SGD update rule, nonconvex use, O(1) cost in m, kernel trick O(m^2).
Built on page 145 (same helpers, same sound pipeline).
Run:   python3 scenes/chapter05/page_150.py            (1080x1080 -> assets/user-data/output)
Quick: P150_RES=540,540 python3 scenes/chapter05/page_150.py
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


class Page150(Page145):
    def construct(self):
        self.setup_stage("oblique", 0.65)
        self._v = "oblique"; self._sub = None
        self.s_update(); self.wipe()
        self.s_nonconvex(); self.wipe()
        self.s_cost(); self.wipe()
        self.s_kernel(); self.wipe()
        self.s_scale(); self.wipe()
        self.wait(0.4)

    def _grid(self, rows, cols, size):
        self.go("top")
        grid = matrix_grid(rows, cols, size, ORIGIN)
        run, lag = 1.5, 0.02
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in grid], lag_ratio=lag), run=run,
                 n=rows * cols, lag=lag, at=0.6)
        self.keep(grid)
        cells = [grid_cell(grid, cols, i, j) for i in range(rows) for j in range(cols)]
        return grid, cells

    # ---- update rule
    def s_update(self):
        self.caption("Follow the estimated gradient downhill")
        self.say(r"$\theta \leftarrow \theta - \epsilon g$, where $\epsilon$ is the learning rate")
        self.go("oblique")
        self.floor(-0.05)
        f = lambda t: 0.38 * t * t - 1.2
        df = lambda t: 0.76 * t
        curve = curve3d(lambda t: np.array([t, f(t), 0.14]), (-3.0, 3.0), K, 5)
        self.act("cut", Create(curve), run=1.2)
        self.keep(curve)
        t = 2.5
        ball = sphere(0.17, np.array([t, f(t), 0.34]))
        self.act("grow", FadeIn(ball), run=0.5)
        self.keep(ball)
        rng = np.random.default_rng(150)
        eps = 0.5
        self.say("Each step moves against a noisy gradient estimate")
        for k in range(5):
            g = df(t) + rng.normal(0, 0.15)
            tn = t - eps * g
            p = np.array([t, f(t), 0.34]); q = np.array([tn, f(tn), 0.34])
            ar = arrow3d(p, q, color=ACCENT)
            self.cue("zap", 0.5)
            self.play(FadeIn(ar), run_time=0.4)
            self.keep(ar)
            self.play(ball.animate.move_to(q), FadeOut(ar), run_time=0.5)
            _forget(self, [ar])
            t = tn
        self.cue("ping", 0.5, i=0)
        self.play(lit(ball), run_time=0.5)
        self.orbit(0.4, run=1.8)

    # ---- nonconvex
    def s_nonconvex(self):
        self.caption("Gradient descent works on nonconvex problems")
        self.say("Once regarded as foolhardy or unprincipled")
        self.go("oblique")
        self.floor(-0.05)
        g = lambda t: 0.15 * t * t + 0.5 * np.sin(1.8 * t)
        dg = lambda t: 0.3 * t + 0.9 * np.cos(1.8 * t)
        curve = curve3d(lambda t: np.array([t, g(t), 0.14]), (-3.2, 3.2), K, 5)
        self.act("cut", Create(curve), run=1.2)
        self.keep(curve)
        t = 2.9
        ball = sphere(0.17, np.array([t, g(t), 0.34]))
        self.act("grow", FadeIn(ball), run=0.5)
        self.keep(ball)
        self.say("It may not reach even a local minimum in reasonable time")
        for k in range(9):
            tn = t - 0.25 * dg(t)
            self.play(ball.animate.move_to(np.array([tn, g(tn), 0.34])), run_time=0.35)
            t = tn
        self.say("But it often finds a very low cost quickly enough to be useful")
        self.cue("ping", 0.5, i=2)
        self.play(lit(ball), run_time=0.5)
        self.orbit(0.4, run=1.8)

    # ---- cost per update independent of m
    def s_cost(self):
        self.caption(r"The cost per update does not depend on $m$")
        self.say("A fixed model size: the same minibatch on a small training set")
        grid, cells = self._grid(4, 6, 0.7)
        rng = np.random.default_rng(1501)
        idx = rng.choice(len(cells), 5, replace=False)
        self.cue("ping", 0.5, i=0)
        self.play(*[lit(cells[i]) for i in idx], run_time=0.6)
        self.wait(0.6)
        self.say("And the same minibatch on a much larger training set")
        self.act("fade", *[FadeOut(grid)], run=0.5)
        _forget(self, [grid])
        grid, cells = self._grid(10, 14, 0.4)
        idx = rng.choice(len(cells), 5, replace=False)
        self.cue("ping", 0.5, i=2)
        self.play(*[lit(cells[i]) for i in idx], run_time=0.6)
        self.wait(0.8)

        self.wipe_stage_only()
        self.go("oblique")
        self.floor(-0.05)
        self.say("Updates needed grow with the training set size")
        err = lambda t: 2.2 * np.exp(-1.1 * (t + 3.0)) - 1.5
        curve = curve3d(lambda t: np.array([t, err(t), 0.14]), (-3.0, 3.0), K, 5)
        self.act("cut", Create(curve), run=1.2)
        self.keep(curve)
        self.label([("test error", np.array([-3.4, 1.1, 0.2])), ("training examples sampled", np.array([2.6, -2.9, 0.2]))])
        self.say("Test error reaches its best before SGD has sampled every example")
        pd = sphere(0.17, np.array([1.2, err(1.2), 0.3]))
        self.act("grow", FadeIn(pd), run=0.5)
        self.keep(pd)
        self.cue("ping", 0.5, i=3)
        self.play(lit(pd), run_time=0.5)
        self.say(r"Asymptotic cost of training with SGD is $O(1)$ in $m$")
        self.orbit(0.4, run=1.8)

    def wipe_stage_only(self):
        items = list(self._items())
        if items:
            self.act("fade", *[FadeOut(m) for m in items], run=0.5)
        self._stage = []

    # ---- kernel trick cost
    def s_kernel(self):
        self.caption("Kernel methods need an m by m matrix")
        self.say(r"The matrix $G_{i,j} = k(x^{(i)}, x^{(j)})$ has $m \times m$ entries")
        grid, cells = self._grid(10, 10, 0.4)
        self.cue("ping", 0.5, i=0)
        self.play(*[lit(c) for c in cells], run_time=1.0)
        self.say(r"Constructing it costs $O(m^{2})$")
        self.wait(0.8)
        self.say("Clearly undesirable for datasets with billions of examples")
        self.wait(0.8)

    # ---- scalability
    def s_scale(self):
        self.caption("Deep learning scales to large datasets")
        self.say("Starting in 2006: better generalization on medium-sized datasets")
        grid, cells = self._grid(4, 6, 0.7)
        self.cue("ping", 0.5, i=0)
        self.play(*[lit(c) for c in cells], run_time=0.8)
        self.say("Then industry interest: a scalable way to train nonlinear models")
        self.act("fade", *[unlit(c) for c in cells], run=0.4)
        rng = np.random.default_rng(1502)
        idx = rng.choice(len(cells), 4, replace=False)
        self.play(*[lit(cells[i]) for i in idx], run_time=0.6)
        self.say("SGD and many enhancements are described further in chapter 8")
        self.cue("finale", 1.0)
        self.wait(1.2)


# ================================================================ pipeline
def main():
    os.chdir(ROOT)
    res = os.environ.get("P150_RES", "1080,1080")
    name = "Page150_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page150_timeline.json"
    wav = ROOT / "media/sfx/Page150.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_150.log")
    print(f"rendering {res} @30fps ...  (progress: bash lib3d/watch_progress.sh /tmp/page_150.log)")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page150"],
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
