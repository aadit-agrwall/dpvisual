"""Page 146 - PCA via SVD (eqs 5.85-5.93).
Layout: caption top, equation band + subtitle band at the bottom (both fixed in frame),
3D objects in the middle zone only. Sync: act() cues, no whoosh on an empty stage.
Run:   python3 scenes/chapter05/page_146.py            (1080x1080 -> assets/user-data/output)
Quick: P146_RES=540,540 python3 scenes/chapter05/page_146.py
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
from lib3d.ml.basics_fit import _stage_floor
from scenes.chapter05.page_143 import SR, write_wav
from scenes.chapter05.page_145 import Page145, build_audio, _cloud, _forget, ZOOM

EQ_Y = 2.7


def R2(a):
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


class Page146(Page145):
    # ------------------------------------------------ equation band (fixed in frame)
    def eq(self, tex, run=0.5):
        m = MathTex(tex, color=K)
        if m.width > 7.4:
            m.scale_to_fit_width(7.4)
        if m.height > 0.8:
            m.scale_to_fit_height(0.8)
        m.move_to(DOWN * EQ_Y)
        self.add_fixed_in_frame_mobjects(m)
        self.cue("say", run)
        if self._eq is not None:
            self.play(FadeOut(self._eq), FadeIn(m), run_time=run)
        else:
            self.play(FadeIn(m), run_time=run)
        self._eq = m

    def wipe(self):
        items = list(self._items())
        for extra in (getattr(self, "_cap", None), self._sub, self._eq):
            if extra is not None:
                items.append(extra)
        if items:
            self.act("wipe", *[FadeOut(m) for m in items], run=0.7)
        self._stage = []; self._cap = None; self._sub = None; self._eq = None

    def pts(self, n=40, s1=1.5, s2=0.5):
        X = _cloud(n, 146, s1, s2, 1.8)
        return X - X.mean(0)

    # ------------------------------------------------ animation
    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None; self._eq = None
        self.s_center(); self.wipe()
        self.s_cov(); self.wipe()
        self.s_pca(); self.wipe()
        self.s_svd(); self.wipe()
        self.s_derive(); self.wipe()
        self.s_diag(); self.wipe()
        self.wait(0.4)

    def axes2(self, L=2.8, z=0.03):
        a = [arrow3d(np.array([-L, 0, z]), np.array([L, 0, z]), color=K),
             arrow3d(np.array([0, -L, z]), np.array([0, L, z]), color=K)]
        self.act("pop", *[FadeIn(m) for m in a], run=0.6)
        self.keep(*a)
        return a

    def s_center(self):
        self.caption(r"Centering the data:  $\mathbb{E}[\mathbf{x}]=\mathbf{0}$")
        self.say("If the mean is not zero, the data can be centered")
        self.go("oblique"); self.floor(-0.05)
        self.axes2()
        X = self.pts(); mu = np.array([1.0, 0.7, 0.0])
        dots = [Dot3D(np.array([x, y, 0.1]) + mu, radius=0.09, color=K) for x, y in X]
        run, lag = 1.4, 0.04
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), run=run, n=len(dots), lag=lag, at=0.6)
        self.keep(*dots)
        mk = sphere(0.2, mu + np.array([0, 0, 0.15]))
        self.act("grow", FadeIn(mk), run=0.5)
        self.keep(mk)
        self.cue("ping", 0.5, i=0)
        self.play(lit(mk), run_time=0.5)
        self.say("Subtract the mean from all examples")
        self.eq(r"\mathbb{E}[\mathbf{x}]=\mathbf{0}")
        self.act("snap", *[d.animate.shift(-mu) for d in dots], mk.animate.shift(-mu), run=1.4)
        self.orbit(0.4, run=1.8)

    def cov_scene_objects(self):
        X = self.pts()
        C = X.T @ X / (len(X) - 1)
        w, V = np.linalg.eigh(C)
        fn = lambda t: np.array([*(V @ (1.3 * np.sqrt(w) * np.array([np.cos(t), np.sin(t)]))), 0.12])
        ell = curve3d(fn, (0, 2 * np.pi), ACCENT, 5)
        dots = [Dot3D(np.array([x, y, 0.1]), radius=0.09, color=K) for x, y in X]
        ang = float(np.arctan2(V[1, 1], V[0, 1]))
        return X, dots, ell, ang

    def s_cov(self):
        self.caption("The unbiased sample covariance matrix")
        self.say("Its shape describes how the data spreads")
        self.go("oblique"); self.floor(-0.05)
        self.axes2()
        X, dots, ell, ang = self.cov_scene_objects()
        run, lag = 1.4, 0.04
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), run=run, n=len(dots), lag=lag, at=0.6)
        self.keep(*dots)
        self.eq(r"\mathrm{Var}[\mathbf{x}]=\frac{1}{m-1}\mathbf{X}^{\top}\mathbf{X}")
        self.say("The ellipse is the covariance: tilted, so not diagonal")
        self.act("cut", Create(ell), run=1.2)
        self.keep(ell)
        self.orbit(0.4, run=1.8)

    def s_pca(self):
        self.caption(r"PCA finds  $\mathbf{z}=\mathbf{W}^{\top}\mathbf{x}$  with diagonal  $\mathrm{Var}[\mathbf{z}]$")
        self.say("Rotate the axes to line up with the ellipse")
        self.go("oblique"); self.floor(-0.05)
        self.axes2()
        X, dots, ell, ang = self.cov_scene_objects()
        run, lag = 1.2, 0.04
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), FadeIn(ell), run=run, n=len(dots), lag=lag, at=0.6)
        self.keep(*dots, ell)
        self.eq(r"\mathbf{z}=\mathbf{W}^{\top}\mathbf{x}")
        Rm = R2(-ang)
        new = [Rm @ x for x in X]
        self.act("rot", Rotate(ell, angle=-ang, axis=OUT, about_point=ORIGIN),
                 *[d.animate.move_to(np.array([p[0], p[1], 0.1])) for d, p in zip(dots, new)], run=2.0)
        self.label([(r"$z_1$", np.array([3.2, 0, 0])), (r"$z_2$", np.array([0, 3.2, 0]))])
        self.say("Now Var[z] is diagonal: no linear correlation")
        self.cue("ping", 0.5, i=0); self.cue("ping", 0.5, i=2)
        self.play(*[Indicate(ell, color=ACCENT, scale_factor=1.0)], run_time=0.6)
        self.say(r"Principal components are eigenvectors of $\mathbf{X}^{\top}\mathbf{X}$")
        self.eq(r"\mathbf{X}^{\top}\mathbf{X}=\mathbf{W}\boldsymbol{\Lambda}\mathbf{W}^{\top}")
        self.orbit(0.4, run=1.8)

    def s_svd(self):
        self.caption(r"Singular value decomposition  $\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{W}^{\top}$")
        self.say("W holds the right singular vectors of X")
        self.go("oblique"); self.floor(-0.05)
        self.axes2()
        n = 28
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        P0 = 1.2 * np.stack([np.cos(ang), np.sin(ang)], 1)
        dots = [Dot3D(np.array([x, y, 0.1]), radius=0.13 if i == 0 else 0.09, color=ACCENT if i == 0 else K)
                for i, (x, y) in enumerate(P0)]
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=0.04), run=1.2, n=n, lag=0.04, at=0.6)
        self.keep(*dots)
        self.eq(r"\mathbf{X}=\mathbf{U}\boldsymbol{\Sigma}\mathbf{W}^{\top}")
        Wt, Sg, Um = R2(np.deg2rad(50)), np.diag([1.9, 0.5]), R2(np.deg2rad(35))
        P1 = P0 @ Wt.T
        self.say(r"$\mathbf{W}^{\top}$ rotates the points")
        self.act("rot", *[d.animate.move_to(np.array([p[0], p[1], 0.1])) for d, p in zip(dots, P1)], run=1.4)
        P2 = P1 @ Sg.T
        self.say(r"$\boldsymbol{\Sigma}$ stretches along the axes")
        self.act("snap", *[d.animate.move_to(np.array([p[0], p[1], 0.1])) for d, p in zip(dots, P2)], run=1.4)
        P3 = P2 @ Um.T
        self.say(r"$\mathbf{U}$ rotates again")
        self.act("rot", *[d.animate.move_to(np.array([p[0], p[1], 0.1])) for d, p in zip(dots, P3)], run=1.4)
        self.orbit(0.4, run=1.8)
        self.say("Recover the eigenvector equation with W as the basis")
        self.eq(r"\mathbf{X}^{\top}\mathbf{X}=\mathbf{W}\boldsymbol{\Sigma}^{2}\mathbf{W}^{\top}")
        self.wait(0.6)

    def s_derive(self):
        self.caption("The SVD shows that PCA gives a diagonal  $\\mathrm{Var}[z]$")
        self.say("Start from the covariance of x")
        self.go("slant"); self.floor(-1.6)
        self.eq(r"\mathrm{Var}[\mathbf{x}]=\frac{1}{m-1}\mathbf{X}^{\top}\mathbf{X}")
        xs = [(k - 2.5) * 0.95 for k in range(6)]
        names = [r"$\mathbf{W}$", r"$\boldsymbol{\Sigma}^{\top}$", r"$\mathbf{U}^{\top}$",
                 r"$\mathbf{U}$", r"$\boldsymbol{\Sigma}$", r"$\mathbf{W}^{\top}$"]
        bx = [box((0.6, 0.6, 0.6), np.array([x, 0, 0])) for x in xs]
        self.say("Substitute the SVD and expand the transpose")
        self.eq(r"=\frac{1}{m-1}\left(\mathbf{U}\boldsymbol{\Sigma}\mathbf{W}^{\top}\right)^{\top}\mathbf{U}\boldsymbol{\Sigma}\mathbf{W}^{\top}")
        self.act("stagger", LaggedStart(*[FadeIn(b) for b in bx], lag_ratio=0.2), run=1.4, n=6, lag=0.2, at=0.6)
        self.keep(*bx)
        self.cue("tick", 0.4)
        lb = tags(self, [(nm, np.array([x, 0, 0.7])) for nm, x in zip(names, xs)], 24)
        self.eq(r"=\frac{1}{m-1}\mathbf{W}\boldsymbol{\Sigma}^{\top}\mathbf{U}^{\top}\mathbf{U}\boldsymbol{\Sigma}\mathbf{W}^{\top}")
        self.say(r"$\mathbf{U}^{\top}\mathbf{U}=\mathbf{I}$ because U is orthogonal")
        self.cue("ping", 0.5, i=0); self.cue("ping", 0.5, i=2)
        self.play(lit(bx[2]), lit(bx[3]), run_time=0.5)
        gone = [bx[2], bx[3], lb[2], lb[3]]
        self.act("fade", *[FadeOut(m) for m in gone], run=0.6)
        _forget(self, gone)
        self.say(r"$\boldsymbol{\Sigma}^{\top}\boldsymbol{\Sigma}$ merges into $\boldsymbol{\Sigma}^{2}$")
        self.cue("zap", 0.5)
        self.play(lit(bx[1]), lit(bx[4]), run_time=0.5)
        gone = [bx[1], bx[4], lb[1], lb[4]]
        self.act("fade", *[FadeOut(m) for m in gone], run=0.5)
        _forget(self, gone)
        sx = (xs[1] + xs[4]) / 2
        s2 = box((0.7, 0.7, 0.7), np.array([sx, 0, 0]))
        self.act("grow", FadeIn(s2), run=0.5)
        self.keep(s2)
        self.label([(r"$\boldsymbol{\Sigma}^{2}$", np.array([sx, 0, 0.75]))])
        self.eq(r"=\frac{1}{m-1}\mathbf{W}\boldsymbol{\Sigma}^{2}\mathbf{W}^{\top}")
        self.cue("ping", 0.5, i=3)
        self.play(lit(s2), run_time=0.5)
        self.orbit(0.4, run=1.8)

    def s_diag(self):
        self.caption("The covariance of  $\\mathbf{z}$  is diagonal")
        self.say(r"$\mathrm{Var}[\mathbf{z}]$ is built from $\mathbf{Z}^{\top}\mathbf{Z}$")
        self.go("top")
        rows = cols = 4
        grid = matrix_grid(rows, cols, 0.8, ORIGIN)
        self.eq(r"\mathrm{Var}[\mathbf{z}]=\frac{1}{m-1}\mathbf{Z}^{\top}\mathbf{Z}")
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in grid], lag_ratio=0.05), run=1.2, n=rows * cols, lag=0.05, at=0.6)
        self.keep(grid)
        self.say("Substitute Z = XW")
        self.eq(r"=\frac{1}{m-1}\mathbf{W}^{\top}\mathbf{X}^{\top}\mathbf{X}\mathbf{W}")
        self.cue("ping", 0.5, i=0)
        self.play(*[lit(c) for c in grid], run_time=0.5)
        self.say("Off-diagonal entries vanish: diagonal, as required")
        off = [grid_cell(grid, cols, i, j) for i in range(rows) for j in range(cols) if i != j]
        self.act("fade", *[c.animate.set_fill(PAPER, opacity=0.0).set_stroke(K, opacity=0.2) for c in off], run=1.2)
        self.cue("finale", 1.0)
        self.play(*[Indicate(grid_cell(grid, cols, i, i), color=ACCENT, scale_factor=1.0) for i in range(rows)], run_time=1.0)
        self.wait(0.8)


# ================================================================ pipeline
def main():
    os.chdir(ROOT)
    res = os.environ.get("P146_RES", "1080,1080")
    name = "Page146_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page146_timeline.json"
    wav = ROOT / "media/sfx/Page146.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_146.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page146"],
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
    subprocess.run(["open", str(final)])
    subprocess.run(["open", str(sheet)])


if __name__ == "__main__":
    main()
