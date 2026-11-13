"""Page 145 - Figure 5.8 + 5.8.1 Principal Components Analysis.
SYNC RULES: sounds are cued by act() at the exact start of the animation they belong to;
camera whoosh only plays when a 3D object is on screen; every sound is clamped to its animation.
Run:   python3 scenes/chapter05/page_145.py            (1080x1080 -> assets/user-data/output)
Quick: P145_RES=540,540 python3 scenes/chapter05/page_145.py
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
from scenes.chapter05.page_143 import (SR, snd_whistle, snd_boing, snd_bloop, snd_squeak, snd_tok,
                                       snd_zip, snd_xylo, snd_ping, snd_pop, snd_plink, snd_whoosh,
                                       snd_swish, snd_caption, snd_tada, write_wav)

SUB_Y = -3.35
ZOOM = 0.65
TH = np.deg2rad(40)                       # direction of greatest variance


def _forget(scene, mobs):
    it = scene._items()
    for m in mobs:
        try:
            it.remove(m)
        except ValueError:
            pass


def _cloud(n=60, seed=145, s1=1.7, s2=0.4, lim=2.3):
    rng = np.random.default_rng(seed)
    Z = np.clip(rng.normal(0, 1, (n, 2)) * np.array([s1, s2]), -lim, lim)
    R = np.array([[np.cos(TH), -np.sin(TH)], [np.sin(TH), np.cos(TH)]])
    return Z @ R.T


class Page145(Stage3D):
    # ------------------------------------------------ timeline + cues
    def _now(self):
        return float(self.time)

    def _vis(self):
        return len(self._items()) > 0

    def cue(self, name, dur=0.0, **kw):
        if not hasattr(self, "_cues"):
            self._cues = []
        d = {"t": self._now(), "name": name, "dur": float(dur)}
        d.update(kw)
        self._cues.append(d)

    def act(self, name, *anims, run=1.0, **kw):
        """cue + play in ONE call, so the sound starts on the exact frame the animation starts."""
        self.cue(name, run, **kw)
        self.play(*anims, run_time=run)

    def play(self, *args, **kw):
        if not getattr(self, "_in_cam", False):
            if not hasattr(self, "_events"):
                self._events = []
            try:
                rt = kw.get("run_time")
                if rt is None:
                    rt = max([getattr(a, "run_time", None) or 1.0 for a in args] or [1.0])
                self._events.append({"t": self._now(), "dur": float(rt), "vis": self._vis(),
                                     "kinds": [type(a).__name__ for a in args]})
            except Exception:
                pass
        return super().play(*args, **kw)

    def move_camera(self, *a, **kw):
        if not hasattr(self, "_events"):
            self._events = []
        self._events.append({"t": self._now(), "dur": float(kw.get("run_time", 1.0)),
                             "vis": self._vis(), "kinds": ["Camera"]})
        self._in_cam = True
        try:
            return super().move_camera(*a, **kw)
        finally:
            self._in_cam = False

    def tear_down(self):
        super().tear_down()
        path = os.environ.get("SFX_TIMELINE", str(ROOT / "media/sfx/Page145_timeline.json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"total": self._now(), "events": getattr(self, "_events", []),
                       "cues": getattr(self, "_cues", [])}, f)

    # ------------------------------------------------ camera (silent + instant when stage is empty)
    def go(self, name):
        if name == self._v:
            return
        if self._vis():
            self.view(name, 1.2)
        else:
            phi, theta = VIEWS[name]
            self.set_camera_orientation(phi=phi, theta=theta, zoom=ZOOM)
        self._v = name

    def orbit(self, angle=PI / 2, run=4.0):
        self._v = None
        super().orbit(angle, run)

    # ------------------------------------------------ text bands (fixed in frame)
    def say(self, text, run=0.4):
        t = Tex(text, color=K, font_size=30)
        if t.width > 7.4:
            t.scale_to_fit_width(7.4)
        t.move_to(DOWN * (-SUB_Y))
        self.add_fixed_in_frame_mobjects(t)
        self.cue("say", run)
        if self._sub is not None:
            self.play(FadeOut(self._sub), FadeIn(t), run_time=run)
        else:
            self.play(FadeIn(t), run_time=run)
        self._sub = t

    def label(self, items):
        self.cue("tick", 0.4)
        tags(self, items, 24)

    def wipe(self):
        items = list(self._items())
        cap = getattr(self, "_cap", None)
        if cap is not None:
            items.append(cap)
        if self._sub is not None:
            items.append(self._sub)
        if items:
            self.act("wipe", *[FadeOut(m) for m in items], run=0.7)
        self._stage = []; self._cap = None; self._sub = None

    def floor(self, z):
        self.cue("pop", 0.5)
        _stage_floor(self, ORIGIN, z=z)

    # ------------------------------------------------ animation
    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None
        self.s_fig(); self.wipe()
        self.s_uncorr(); self.wipe()
        self.s_orth(); self.wipe()
        self.s_best(); self.wipe()
        self.s_design(); self.wipe()
        self.wait(0.4)

    def s_fig(self):
        self.caption("Figure 5.8: PCA aligns the greatest variance with the axes")
        self.say("Left: the original data x varies along a diagonal")
        self.go("oblique")
        X = _cloud()
        cL, cR = np.array([-2.9, 0.0, 0.0]), np.array([2.9, 0.0, 0.0])
        h = 2.5

        def frame(c):
            p = [c + np.array([-h, -h, 0.02]), c + np.array([h, -h, 0.02]),
                 c + np.array([h, h, 0.02]), c + np.array([-h, h, 0.02])]
            return [segment(p[k], p[(k + 1) % 4], width=5) for k in range(4)]

        def axes(c):
            return [arrow3d(c + np.array([-2.2, 0, 0.03]), c + np.array([2.2, 0, 0.03]), color=K),
                    arrow3d(c + np.array([0, -2.2, 0.03]), c + np.array([0, 2.2, 0.03]), color=K)]

        fr = frame(cL) + frame(cR)
        self.act("cut", *[Create(s) for s in fr], run=1.0)
        self.keep(*fr)
        ax = axes(cL) + axes(cR)
        self.act("pop", *[FadeIn(a) for a in ax], run=0.6)
        self.keep(*ax)
        self.label([(r"$x_1$", cL + np.array([2.0, -2.85, 0])), (r"$x_2$", cL + np.array([-2.85, 2.0, 0])),
                    (r"$z_1$", cR + np.array([2.0, -2.85, 0])), (r"$z_2$", cR + np.array([-2.85, 2.0, 0]))])

        dl = [Dot3D(cL + np.array([x, y, 0.1]), radius=0.09, color=K) for x, y in X]
        run, lag = 1.6, 0.03
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dl], lag_ratio=lag), run=run, n=len(dl), lag=lag, at=0.6)
        self.keep(*dl)
        u = np.array([np.cos(TH), np.sin(TH), 0.0]); v = np.array([-np.sin(TH), np.cos(TH), 0.0])
        z = np.array([0, 0, 0.12])
        al = [arrow3d(cL + z, cL + z + u * 2.3, color=ACCENT), arrow3d(cL + z, cL + z + v * 1.1, color=ACCENT)]
        self.act("zap", *[FadeIn(a) for a in al], run=0.7)
        self.keep(*al)
        self.orbit(0.35, run=1.6)

        self.say(r"Right: $z = x^{\top} W$, the data varies most along $z_1$")
        dr = [Dot3D(cR + np.array([x, y, 0.1]), radius=0.09, color=K) for x, y in X]
        ar = [arrow3d(cR + z, cR + z + u * 2.3, color=ACCENT), arrow3d(cR + z, cR + z + v * 1.1, color=ACCENT)]
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dr], lag_ratio=lag), FadeIn(ar[0]), FadeIn(ar[1]),
                 run=run, n=len(dr), lag=lag, at=0.6)
        self.keep(*dr, *ar)
        grp = VGroup(*dr, *ar)
        self.act("rot", Rotate(grp, angle=-TH, axis=OUT, about_point=cR), run=2.0)
        self.say(r"The second-most variance is now along $z_2$")
        self.cue("ping", 0.5, i=0); self.cue("ping", 0.5, i=2)
        self.play(*[Indicate(a, color=ACCENT, scale_factor=1.0) for a in ar[1:]], run_time=0.6)
        self.orbit(0.35, run=1.6)

    def s_uncorr(self):
        self.caption("PCA removes linear correlation, not all dependence")
        self.say("Lower dimensionality, and elements with no linear correlation")
        self.go("oblique")
        self.floor(-0.05)
        rng = np.random.default_rng(7)
        xs = np.linspace(-2.3, 2.3, 22)
        ys = 0.38 * xs ** 2 - 1.2 + rng.normal(0, 0.08, len(xs))
        dots = [Dot3D(np.array([x, y, 0.1]), radius=0.1, color=K) for x, y in zip(xs, ys)]
        run, lag = 1.6, 0.08
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), run=run, n=len(dots), lag=lag, at=0.6)
        self.keep(*dots)
        ln = Line(np.array([-2.7, float(ys.mean()), 0.12]), np.array([2.7, float(ys.mean()), 0.12]),
                  color=ACCENT, stroke_width=5)
        self.say("A flat line fits: zero linear correlation")
        self.act("cut", Create(ln), run=0.8)
        self.keep(ln)
        self.orbit(0.35, run=1.6)
        self.say("Yet y still depends on x: independence needs nonlinear removal")
        par = curve3d(lambda t: np.array([t, 0.38 * t * t - 1.2, 0.14]), (-2.4, 2.4), ACCENT, 5)
        self.act("cut", Create(par), run=1.2)
        self.keep(par)
        self.orbit(0.35, run=1.6)

    def s_orth(self):
        self.caption("PCA: an orthogonal, linear transformation  $z = x^{\\top} W$")
        self.say("Input x lives in the original axes")
        self.go("oblique")
        self.floor(-2.5)
        L = 2.4
        E = [np.array(e, dtype=float) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        black = [arrow3d(np.zeros(3), L * e, color=K) for e in E]
        self.act("pop", *[FadeIn(a) for a in black], run=0.7)
        self.keep(*black)
        p = np.array([1.2, 0.9, 1.0])
        pd = sphere(0.16, p)
        self.act("grow", FadeIn(pd), run=0.5)
        self.keep(pd)
        self.say("W rotates to new axes that stay orthogonal")
        blue = VGroup(*[arrow3d(np.zeros(3), L * e, color=ACCENT) for e in E])
        self.act("zap", FadeIn(blue), run=0.6)
        self.keep(blue)
        a, b = np.deg2rad(35), np.deg2rad(25)
        self.act("rot", Rotate(blue, angle=a, axis=OUT, about_point=ORIGIN), run=1.4)
        self.act("rot", Rotate(blue, angle=b, axis=RIGHT, about_point=ORIGIN), run=1.4)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        u1 = (Rx @ Rz)[:, 0]
        proj = float(p @ u1) * u1
        self.say("z is x expressed in the new axes")
        seg = segment(p, proj, width=4, color=ACCENT)
        self.act("cut", Create(seg), run=0.8)
        self.keep(seg)
        self.cue("ping", 0.5, i=3)
        self.play(lit(pd), run_time=0.5)
        self.orbit(0.5, run=2.0)

    def s_best(self):
        self.caption("The first principal component reconstructs best")
        self.say("Project each point onto a 1-D line")
        self.go("oblique")
        self.floor(-0.05)
        X = _cloud(30, 11, 1.6, 0.45, 2.2)
        dots = [Dot3D(np.array([x, y, 0.1]), radius=0.1, color=K) for x, y in X]
        run, lag = 1.3, 0.05
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), run=run, n=len(dots), lag=lag, at=0.6)
        self.keep(*dots)

        def line_group(deg):
            d = np.array([np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))])
            err = 0.0
            res = []
            for x in X:
                pr = float(x @ d) * d
                err += float(np.sum((x - pr) ** 2))
                res.append(Line(np.array([x[0], x[1], 0.12]), np.array([pr[0], pr[1], 0.12]), color=K, stroke_width=2.5))
            ln = Line(np.array([*(-3.0 * d), 0.12]), np.array([*(3.0 * d), 0.12]), color=ACCENT, stroke_width=6)
            return VGroup(ln, *res), err / len(X)

        prev = None
        notes = ["Poor line: large reconstruction error", "Better line: error shrinks",
                 "First principal component: least-squares error is smallest"]
        for k, deg in enumerate((115, 75, 40)):
            g, err = line_group(deg)
            if prev is not None:
                self.act("fade", FadeOut(prev), run=0.4)
                _forget(self, [prev])
            self.say(notes[k])
            self.act("cut", Create(g), run=0.9)
            self.keep(g)
            self.wait(0.3)
            prev = g
        self.cue("ping", 0.5, i=4)
        self.play(*[m.animate.set_color(ACCENT) for m in prev[1:]], run_time=0.5)
        self.orbit(0.4, run=1.8)

    def s_design(self):
        self.caption("The  $m \\times n$  design matrix  $X$")
        self.say("Each row is one example, each column one feature")
        self.go("top")
        rows, cols = 6, 3
        grid = matrix_grid(rows, cols, 0.8, ORIGIN)
        n = rows * cols
        run, lag = 1.6, 0.06
        self.act("stagger", LaggedStart(*[FadeIn(c) for c in grid], lag_ratio=lag), run=run, n=n, lag=lag, at=0.6)
        self.keep(grid)
        r = [grid_cell(grid, cols, 2, j) for j in range(cols)]
        self.cue("ping", 0.5, i=0)
        self.play(*[lit(c) for c in r], run_time=0.5)
        self.say("A row: one example x")
        self.wait(0.4)
        self.act("fade", *[unlit(c) for c in r], run=0.4)
        c1 = [grid_cell(grid, cols, i, 1) for i in range(rows)]
        self.say("A column: one feature across all examples")
        self.cue("ping", 0.5, i=2)
        self.play(*[lit(c) for c in c1], run_time=0.5)
        self.wait(0.4)
        self.say("Next: how the PCA representation decorrelates X")
        self.cue("finale", 1.0)
        self.play(*[unlit(c) for c in c1], run_time=0.5)
        self.wait(0.8)


# ================================================================ sound
def _fit(s, maxd):
    n = max(int(maxd * SR), 2)
    if len(s) > n:
        s = s[:n].copy()
        k = min(int(0.04 * SR), n)
        s[-k:] *= np.linspace(1, 0, k)
    return s


def build_audio(data):
    total = data["total"]
    buf = np.zeros(int((total + 2.0) * SR))
    log = []

    def add(name, t, s, maxd):
        s = _fit(s, maxd)
        log.append((round(t, 2), name))
        i = int(t * SR)
        if 0 <= i < len(buf):
            j = min(len(buf), i + len(s)); buf[i:j] += s[:j - i]

    def stag(c, fn):
        n, lag, at, d = c["n"], c["lag"], c["at"], max(c["dur"], 0.05)
        unit = d / (1 + (n - 1) * lag)                       # Manim LaggedStart: real per-item length
        idx = np.unique(np.linspace(0, n - 1, min(n, 14)).astype(int))
        for k, i in enumerate(idx):
            fn(k, c["t"] + i * lag * unit + at * unit, unit * 1.2 + 0.15)

    for e in data["events"]:
        if "Camera" in e["kinds"]:
            if e["vis"] and e["dur"] >= 0.6:                 # never whoosh over an empty stage
                add("whoosh", e["t"], snd_whoosh(e["dur"]), e["dur"] + 0.1)
        elif "Write" in e["kinds"]:
            add("caption-kazoo", e["t"], snd_caption(e["dur"]), e["dur"] + 0.3)

    for c in data["cues"]:
        nm, t, d = c["name"], c["t"], max(c["dur"], 0.05)
        mx = d + 0.25
        if nm == "pop":
            add("pop", t, snd_pop(1.0, 0.35), mx)
        elif nm == "say":
            add("say-tok", t, snd_tok(0.13, 900), 0.2); add("say-pop", t + 0.05, snd_pop(1.6, 0.12), 0.2)
        elif nm == "tick":
            for k in range(4):
                add("label-tick", t + k * d / 4, snd_tok(0.12, 900 + 60 * k), 0.15)
        elif nm == "grow":
            add("grow", t, snd_whistle(220, 600, min(d, 0.22), 0.28), mx); add("squeak", t + 0.1, snd_squeak(1.1, 0.14), 0.15)
        elif nm == "stagger":
            stag(c, lambda k, ti, m: add("bloop", ti, snd_bloop(1 + 0.03 * k, 0.3), m))
        elif nm == "rot":
            add("rot-whistle", t, snd_whistle(300, 800, d * 0.9, 0.22), d)
            add("boing", t + d * 0.85, snd_boing(0.25), d * 0.15 + 0.2)
        elif nm == "cut":
            add("zip", t, snd_zip(d), d); add("cut-tok", t + max(d - 0.06, 0), snd_tok(0.3, 560), 0.15)
        elif nm == "zap":
            add("zap-xylo", t, snd_xylo(5, 0.28), mx); add("zap-whistle", t, snd_whistle(500, 1100, 0.2, 0.18), mx)
        elif nm == "ping":
            add("ping", t, snd_ping(c["i"], 0.4 if c["i"] < 3 else 0.3), 0.5)
        elif nm == "fade":
            add("swish", t, snd_swish(0.45), mx)
        elif nm == "wipe":
            add("wipe-whistle", t, snd_whistle(1500, 350, min(d, 0.4), 0.2), d)
            add("swish", t, snd_swish(0.5), d)
        elif nm == "finale":
            add("TA-DA", t, snd_tada(0.3), 1.4)

    buf = buf[:int((total + 0.3) * SR)]
    buf *= 0.89 / (np.abs(buf).max() + 1e-9)
    buf = np.tanh(buf * 1.15) / np.tanh(1.15)
    return buf, log


# ================================================================ pipeline
def main():
    os.chdir(ROOT)
    res = os.environ.get("P145_RES", "1080,1080")
    name = "Page145_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page145_timeline.json"
    wav = ROOT / "media/sfx/Page145.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_145.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page145"],
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
