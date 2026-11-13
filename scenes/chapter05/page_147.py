"""Page 147 - end of 5.8.1 PCA (diagonal covariance) + 5.8.2 k-means Clustering.
Built on page 145 (same helpers, same sound pipeline).
Run:   python3 scenes/chapter05/page_147.py            (1080x1080 -> assets/user-data/output)
Quick: P147_RES=540,540 python3 scenes/chapter05/page_147.py
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
from scenes.chapter05.page_145 import Page145, build_audio, _forget, _cloud, TH


class Page147(Page145):
    def tear_down(self):
        super().tear_down()

    def construct(self):
        self.setup_stage("oblique", 0.65)
        self._v = "oblique"; self._sub = None
        self.s_decor(); self.wipe()
        self.s_cov(); self.wipe()
        self.s_beyond(); self.wipe()
        self.s_kmeans(); self.wipe()
        self.s_onehot(); self.wipe()
        self.wait(0.4)

    # ---- PCA: rotation that decorrelates
    def s_decor(self):
        self.caption("PCA: a rotation that decorrelates the data")
        self.say(r"$z = x^{\top} W$ rotates the data onto new axes")
        self.go("oblique")
        self.floor(-0.05)
        X = _cloud(50, 147, 1.7, 0.45, 2.3)
        u = np.array([np.cos(TH), np.sin(TH), 0.0]); v = np.array([-np.sin(TH), np.cos(TH), 0.0])
        z = np.array([0, 0, 0.12])
        ax = [arrow3d(np.array([-2.8, 0, 0.03]), np.array([2.8, 0, 0.03]), color=K),
              arrow3d(np.array([0, -2.8, 0.03]), np.array([0, 2.8, 0.03]), color=K)]
        self.act("pop", *[FadeIn(a) for a in ax], run=0.6)
        self.keep(*ax)
        self.label([(r"$z_1$", np.array([2.9, -0.4, 0])), (r"$z_2$", np.array([-0.4, 2.9, 0]))])
        dl = [Dot3D(np.array([x, y, 0.1]), radius=0.09, color=K) for x, y in X]
        run, lag = 1.5, 0.03
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dl], lag_ratio=lag), run=run, n=len(dl), lag=lag, at=0.6)
        self.keep(*dl)
        al = [arrow3d(z, z + u * 2.4, color=ACCENT), arrow3d(z, z + v * 1.2, color=ACCENT)]
        self.act("zap", *[FadeIn(a) for a in al], run=0.7)
        self.keep(*al)
        self.say("Principal axes of variance become the new basis")
        grp = VGroup(*dl, *al)
        self.act("rot", Rotate(grp, angle=-TH, axis=OUT, about_point=ORIGIN), run=2.2)
        self.say("The elements of z are mutually uncorrelated")
        self.cue("ping", 0.5, i=0); self.cue("ping", 0.5, i=2)
        self.play(*[Indicate(a, color=ACCENT, scale_factor=1.0) for a in al], run_time=0.6)
        self.orbit(0.35, run=1.8)

    # ---- diagonal covariance matrix
    def s_cov(self):
        self.caption(r"Covariance of $z$ is diagonal")
        self.say(r"Using $W^{\top}W = I$, the covariance is $\frac{1}{m-1}\Sigma^{2}$")
        self.go("top")
        n = 4
        grid = matrix_grid(n, n, 0.8, ORIGIN)
        run, lag = 1.6, 0.06
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in grid], lag_ratio=lag), run=run,
                 n=n * n, lag=lag, at=0.6)
        self.keep(grid)
        diag = [grid_cell(grid, n, i, i) for i in range(n)]
        self.say("Only the diagonal is nonzero: variances along each axis")
        self.cue("ping", 0.5, i=3)
        self.play(*[lit(c) for c in diag], run_time=0.6)
        self.label([(r"$\sigma_%d^2$" % (i + 1), c.get_center() + np.array([0, 0, 0.05]))
                    for i, c in enumerate(diag)])
        self.wait(0.6)
        self.say("Off-diagonal entries are zero: no linear correlation")
        self.wait(0.6)

    # ---- correlation is not all dependence
    def s_beyond(self):
        self.caption("PCA disentangles only linear dependence")
        self.say("A rotation aligns axes with the factors of variation")
        self.go("oblique")
        self.floor(-0.05)
        rng = np.random.default_rng(147)
        xs = np.linspace(-2.3, 2.3, 22)
        ys = 0.38 * xs ** 2 - 1.2 + rng.normal(0, 0.08, len(xs))
        dots = [Dot3D(np.array([x, y, 0.1]), radius=0.1, color=K) for x, y in zip(xs, ys)]
        run, lag = 1.5, 0.07
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), run=run, n=len(dots), lag=lag, at=0.6)
        self.keep(*dots)
        self.say("Complicated feature dependencies need more than a linear map")
        par = curve3d(lambda t: np.array([t, 0.38 * t * t - 1.2, 0.14]), (-2.4, 2.4), ACCENT, 5)
        self.act("cut", Create(par), run=1.2)
        self.keep(par)
        self.orbit(0.4, run=1.8)

    # ---- 5.8.2 k-means
    def s_kmeans(self):
        self.caption("5.8.2  k-means clustering")
        self.say("Divide the training set into k clusters of nearby examples")
        self.go("oblique")
        self.floor(-0.05)
        rng = np.random.default_rng(1472)
        centers = [np.array([-2.6, -1.0]), np.array([2.4, -1.4]), np.array([0.2, 2.4])]
        clusters = []
        for c in centers:
            pts = c + rng.normal(size=(14, 2)) * 0.5
            clusters.append([Dot3D(np.array([x, y, 0.1]), radius=0.1, color=K) for x, y in pts])
        alld = [d for g in clusters for d in g]
        run, lag = 1.6, 0.03
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in alld], lag_ratio=lag), run=run, n=len(alld), lag=lag, at=0.6)
        self.keep(*alld)
        codes = [r"$h=(1,0,0)$", r"$h=(0,1,0)$", r"$h=(0,0,1)$"]
        self.say(r"$x$ in cluster $i$ gives $h_i = 1$, all other entries $0$")
        prev = None
        for k, c in enumerate(centers):
            if prev is not None:
                self.play(*[d.animate.set_color(K) for d in prev], run_time=0.3)
            self.cue("ping", 0.5, i=k)
            self.play(*[d.animate.set_color(ACCENT) for d in clusters[k]], run_time=0.5)
            self.label([(codes[k], np.array([c[0], c[1] + 1.2, 0.2]))])
            self.wait(0.5)
            prev = clusters[k]
        self.orbit(0.4, run=1.8)

    # ---- one-hot is sparse
    def s_onehot(self):
        self.caption("A one-hot code is a sparse representation")
        self.say("Most entries of h are zero for every input")
        self.go("top")
        k = 8
        ga = matrix_grid(1, k, 0.7, UP * 1.2)
        gb = matrix_grid(1, k, 0.7, DOWN * 1.2)
        run, lag = 1.6, 0.04
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in list(ga) + list(gb)], lag_ratio=lag),
                 run=run, n=2 * k, lag=lag, at=0.6)
        self.keep(ga, gb)
        ca = [grid_cell(ga, k, 0, j) for j in range(k)]
        hot = 3
        self.label([(("1" if j == hot else "0"), c.get_center() + np.array([0, 0, 0.05]))
                    for j, c in enumerate(ca)])
        self.cue("ping", 0.5, i=0)
        self.play(lit(ca[hot]), run_time=0.5)
        self.say("One-hot: statistically similar within a cluster, cheap to compute")
        self.wait(0.8)
        cb = [grid_cell(gb, k, 0, j) for j in range(k)]
        self.say("Later: sparse codes with more than one nonzero entry")
        self.cue("ping", 0.5, i=2)
        self.play(*[lit(cb[j]) for j in (0, 3, 5)], run_time=0.6)
        self.cue("finale", 1.0)
        self.wait(1.0)


# ================================================================ pipeline
def main():
    os.chdir(ROOT)
    res = os.environ.get("P147_RES", "1080,1080")
    name = "Page147_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page147_timeline.json"
    wav = ROOT / "media/sfx/Page147.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_147.log")
    print(f"rendering {res} @30fps ...  (progress: tail -f {log})")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page147"],
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
