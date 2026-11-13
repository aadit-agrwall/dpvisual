"""Page 144 - unsupervised learning: representations (lower-dimensional, sparse, independent).
Layout rule: ALL sentences live in fixed screen bands (caption top, subtitle bottom);
3D objects stay in the middle zone, so text never overlaps objects.
Run:   python3 scenes/chapter05/page_144.py            (1080x1080 -> assets/user-data/output)
Quick: P144_RES=540,540 python3 scenes/chapter05/page_144.py
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

SUB_Y = -3.35          # subtitle band (fixed in frame)
ZOOM = 0.65            # keeps every 3D object inside the middle zone


def _forget(scene, mobs):
    it = scene._items()
    for m in mobs:
        try:
            it.remove(m)
        except ValueError:
            pass


class Page144(Stage3D):
    # ---- cue + timeline recording (keeps sound in sync)
    def _now(self):
        return float(self.time)

    def cue(self, name, dur=0.0, dt=0.0, **kw):
        if not hasattr(self, "_cues"):
            self._cues = []
        d = {"t": self._now() + dt, "name": name, "dur": float(dur)}
        d.update(kw)
        self._cues.append(d)

    def play(self, *args, **kw):
        if not getattr(self, "_in_cam", False):
            if not hasattr(self, "_events"):
                self._events = []
            try:
                rt = kw.get("run_time")
                if rt is None:
                    rt = max([getattr(a, "run_time", None) or 1.0 for a in args] or [1.0])
                self._events.append({"t": self._now(), "dur": float(rt),
                                     "kinds": [type(a).__name__ for a in args]})
            except Exception:
                pass
        return super().play(*args, **kw)

    def move_camera(self, *a, **kw):
        if not hasattr(self, "_events"):
            self._events = []
        self._events.append({"t": self._now(), "dur": float(kw.get("run_time", 1.0)), "kinds": ["Camera"]})
        self._in_cam = True
        try:
            return super().move_camera(*a, **kw)
        finally:
            self._in_cam = False

    def tear_down(self):
        super().tear_down()
        path = os.environ.get("SFX_TIMELINE", str(ROOT / "media/sfx/Page144_timeline.json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"total": self._now(), "events": getattr(self, "_events", []),
                       "cues": getattr(self, "_cues", [])}, f)

    # ---- camera: move only when the view really changes (orbit makes it dirty)
    def go(self, name):
        if name != self._v:
            self.view(name, 1.2)
            self._v = name

    def orbit(self, angle=PI / 2, run=4.0):
        self._v = None
        super().orbit(angle, run)

    # ---- text: subtitle band (fixed in frame, never inside the 3D zone)
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

    def wipe(self):
        self.cue("wipe", 0.7)
        items = list(self._items())
        cap = getattr(self, "_cap", None)
        if cap is not None:
            items.append(cap)
        if self._sub is not None:
            items.append(self._sub)
        if items:
            self.play(*[FadeOut(m) for m in items], run_time=0.7)
        self._stage = []; self._cap = None; self._sub = None

    # ---- animation
    def construct(self):
        self.setup_stage("oblique", ZOOM)
        self._v = "oblique"; self._sub = None
        self.s_tasks(); self.wipe()
        self.s_denoise(); self.wipe()
        self.s_cluster(); self.wipe()
        self.s_lowdim(); self.wipe()
        self.s_sparse(); self.wipe()
        self.s_indep(); self.wipe()
        self.s_venn(); self.wipe()
        self.s_theme(); self.wipe()
        self.wait(0.4)

    def floor(self, z):
        self.cue("pop", 0.5)
        _stage_floor(self, ORIGIN, z=z)

    def s_tasks(self):
        self.caption("5.8  Unsupervised learning: find structure")
        self.say("Density estimation: learn the distribution")
        self.go("oblique")
        self.floor(-0.05)
        f = lambda u, v: 2.2 * np.exp(-(u * u + v * v) / (2 * 1.1 ** 2))
        surf = surface(f, (-3, 3), (-3, 3), (24, 24))
        self.cue("build", 1.5)
        self.play(Create(surf), run_time=1.5)
        self.keep(surf)
        self.wait(0.4)
        self.say("Draw samples: new points follow that density")
        rng = np.random.default_rng(3)
        xy = np.clip(rng.normal(0, 1.1, (14, 2)), -2.6, 2.6)
        dots = [Dot3D(np.array([x, y, float(f(x, y)) + 0.12]), radius=0.1, color=K) for x, y in xy]
        run, lag = 2.0, 0.25
        self.cue("drop", run, lag=lag, n=len(dots), at=0.7)
        self.play(LaggedStart(*[FadeIn(d, shift=IN * 1.6) for d in dots], lag_ratio=lag), run_time=run)
        self.keep(*dots)
        self.orbit(0.5, run=2.0)
        self.say("No labels needed: the data itself is the signal")
        self.wait(0.6)

    def s_denoise(self):
        self.caption("Denoise, or find the manifold the data lies near")
        self.say("Real data is noisy")
        self.go("oblique")
        self.floor(-1.6)
        T = 3 * np.pi
        f = lambda t: np.array([(0.4 + 0.18 * t) * np.cos(t), (0.4 + 0.18 * t) * np.sin(t), 0.18 * t - 1.0])
        ts = np.linspace(0.4, T - 0.3, 26)
        on = [f(t) for t in ts]
        rng = np.random.default_rng(9)
        noisy = [p + np.clip(rng.normal(0, 0.35, 3), -0.6, 0.6) for p in on]
        dots = [Dot3D(p, radius=0.09, color=K) for p in noisy]
        run, lag = 1.6, 0.1
        self.cue("stagger", run, lag=lag, n=len(dots), at=0.6)
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), run_time=run)
        self.keep(*dots)
        self.orbit(0.4, run=1.6)
        self.say("Denoise: pull each point back onto the structure")
        self.cue("snap", 1.4)
        self.play(*[d.animate.move_to(p) for d, p in zip(dots, on)], run_time=1.4)
        curve = curve3d(f, (0, T), ACCENT, 4)
        self.cue("cut", 1.4)
        self.play(Create(curve), run_time=1.4)
        self.keep(curve)
        self.say("A manifold: the data lies near a low-dimensional curve")
        self.orbit(0.5, run=2.0)

    def s_cluster(self):
        self.caption("Clustering: group related examples")
        self.say("k-means: every point joins its nearest center")
        self.go("oblique")
        self.floor(-2.0)
        km = KMeans3D()
        self.cue("pop", 1.0)
        self.play(FadeIn(km.dots), run_time=1.0)
        self.cue("grow", 0.6)
        self.play(*[GrowFromCenter(c) for c in km.cents], run_time=0.6)
        self.keep(km.group)
        for i in range(min(2, len(km.steps))):
            self.cue("cut", 0.6); self.cue("snap", 0.8, dt=0.6); self.cue("fade", 0.4, dt=1.4)
            km.step(self, i, 1.0)
        self.orbit(0.5, run=2.0)
        self.say("Groups of related examples emerge")
        self.wait(0.5)

    def s_lowdim(self):
        self.caption("Lower-dimensional representation")
        self.say("Compress x into fewer numbers")
        self.go("oblique")
        self.floor(-2.5)
        rng = np.random.default_rng(5)
        n = 36
        a, b = np.deg2rad(35), np.deg2rad(30)
        Rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        R = Rz @ Rx
        X = (np.clip(rng.normal(0, 1, (n, 3)), -2.2, 2.2) * np.array([1.15, 0.6, 0.12])) @ R.T
        U, V, N = R[:, 0], R[:, 1], R[:, 2]
        dots = [Dot3D(x, radius=0.08, color=K) for x in X]
        run, lag = 1.2, 0.05
        self.cue("stagger", run, lag=lag, n=n, at=0.6)
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), run_time=run)
        self.keep(*dots)
        a1 = arrow3d(np.zeros(3), U * 2.6, color=ACCENT)
        a2 = arrow3d(np.zeros(3), V * 1.6, color=ACCENT)
        self.cue("cut", 0.8)
        self.play(FadeIn(a1), FadeIn(a2), run_time=0.8)
        self.keep(a1, a2)
        self.say("Keep the directions that carry the most information")
        plane = plane_poly(np.zeros(3), N, 5.0, 0.15)
        self.cue("pop", 0.6)
        self.play(FadeIn(plane), run_time=0.6)
        self.keep(plane)
        self.orbit(0.4, run=1.8)
        P = X - np.outer(X @ N, N)
        self.cue("squash", 1.4)
        self.play(*[d.animate.move_to(p) for d, p in zip(dots, P)], run_time=1.4)
        self.say("Two numbers per point instead of three")
        self.orbit(0.4, run=1.8)

    def s_sparse(self):
        self.caption("Sparse representation")
        self.say("Higher-dimensional, but mostly zeros")
        self.go("slant")
        self.floor(-2.2)
        xin = np.linspace(-0.9, 0.9, 4)
        xout = np.linspace(-3.3, 3.3, 12)
        inp = [box((0.4, 0.4, 0.4), np.array([x, 0, 1.2]), sw=2) for x in xin]
        rep = [box((0.4, 0.4, 0.4), np.array([x, 0, -1.0]), sw=2) for x in xout]
        self.cue("stagger", 0.8, lag=0.3, n=4, at=0.6)
        self.play(LaggedStart(*[FadeIn(m) for m in inp], lag_ratio=0.3), run_time=0.8)
        self.cue("stagger", 1.4, lag=0.15, n=12, at=0.6)
        self.play(LaggedStart(*[FadeIn(m) for m in rep], lag_ratio=0.15), run_time=1.4)
        self.keep(*inp, *rep)
        on = [3, 8]
        self.cue("zap", 0.7)
        flashes(self, [(np.array([0, 0, 1.0]), np.array([xout[i], 0, -0.8])) for i in on], run=0.7)
        self.cue("ping", 0.5, i=0); self.cue("ping", 0.5, i=2, dt=0.1)
        self.play(*[lit(rep[i]) for i in on], run_time=0.5)
        self.orbit(0.4, run=1.8)
        self.say("Most entries are zero, yet little information is lost")
        self.wait(0.6)

    def s_indep(self):
        self.caption("Independent representation")
        self.say("Disentangle the sources of variation")
        self.go("oblique")
        self.floor(-0.05)
        rng = np.random.default_rng(12)
        B = np.clip(rng.normal(0, 1, (24, 2)), -1.8, 1.8) * np.array([1.4, 0.5])
        a = np.deg2rad(40); c, s = np.cos(a), np.sin(a)
        Tl = B @ np.array([[c, -s], [s, c]]).T
        dots = [Dot3D(np.array([x, y, 0.1]), radius=0.1, color=K) for x, y in Tl]
        ax1 = arrow3d(np.array([-3.0, 0, 0.05]), np.array([3.2, 0, 0.05]), color=K)
        ax2 = arrow3d(np.array([0, -3.0, 0.05]), np.array([0, 3.2, 0.05]), color=K)
        run, lag = 1.4, 0.06
        self.cue("pop", 0.8)
        self.play(FadeIn(ax1), FadeIn(ax2), run_time=0.8)
        self.cue("stagger", run, lag=lag, n=len(dots), at=0.6)
        self.play(LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), run_time=run)
        line = Line(np.array([-2.7 * c, -2.7 * s, 0.12]), np.array([2.7 * c, 2.7 * s, 0.12]),
                    color=ACCENT, stroke_width=5)
        self.cue("cut", 0.7)
        self.play(Create(line), run_time=0.7)
        self.keep(ax1, ax2, *dots, line)
        self.say("Dependent: the two coordinates move together")
        self.orbit(0.4, run=1.8)
        self.say("Independent: each axis varies on its own")
        self.cue("snap", 1.6)
        self.play(*[d.animate.move_to(np.array([bx, by, 0.1])) for d, (bx, by) in zip(dots, B)],
                  line.animate.put_start_and_end_on(np.array([-2.7, 0, 0.12]), np.array([2.7, 0, 0.12])),
                  run_time=1.6)
        self.orbit(0.4, run=1.8)

    def s_venn(self):
        self.caption("The three criteria are not mutually exclusive")
        self.say("Three ways to build a simpler representation")
        self.go("top")
        r = 1.35
        C = [(-0.9, 0.7), (0.9, 0.7), (0.0, -0.85)]
        circ = [curve3d(lambda t, c=c: np.array([c[0] + r * np.cos(t), c[1] + r * np.sin(t), 0.03]),
                        (0, 2 * np.pi), K, 5) for c in C]
        for m in circ:
            self.cue("cut", 0.6)
            self.play(Create(m), run_time=0.6)
            self.keep(m)
        names = ["Lower-dimensional", "Sparse", "Independent"]
        for i, m in enumerate(circ):
            self.say(names[i])
            self.cue("ping", 0.5, i=i)
            self.play(m.animate.set_color(ACCENT), run_time=0.5)
            self.wait(0.3)
            self.play(m.animate.set_color(K), run_time=0.4)
        core = sphere(0.2, np.array([0.0, 0.2, 0.1]))
        self.say("They overlap: removing redundancy helps all three")
        self.cue("pop", 0.6)
        self.play(FadeIn(core), run_time=0.6)
        self.keep(core)
        self.cue("ping", 0.5, i=4)
        self.play(lit(core), *[m.animate.set_color(ACCENT) for m in circ], run_time=0.6)
        self.wait(0.8)

    def s_theme(self):
        self.caption("Representation: a central theme of deep learning")
        self.say("Learn a simpler, more useful form of the input")
        self.go("slant")
        self.floor(-2.0)
        bx = box((1.0, 1.0, 1.0), np.array([-2.6, 0, 0]))
        bh = box((0.6, 0.6, 0.6), np.array([2.6, 0, 0]))
        ln = segment(np.array([-2.0, 0, 0]), np.array([2.1, 0, 0]), width=4)
        self.cue("pop", 0.8)
        self.play(FadeIn(bx), FadeIn(bh), Create(ln), run_time=0.8)
        self.keep(bx, bh, ln)
        self.say("Input x becomes representation h")
        self.cue("ping", 0.4, i=0)
        self.play(lit(bx), run_time=0.4)
        self.cue("zap", 0.8)
        flashes(self, [(np.array([-2.0, 0, 0]), np.array([2.1, 0, 0]))], run=0.8)
        self.cue("ping", 0.4, i=2)
        self.play(lit(bh), run_time=0.4)
        self.cue("finale", 1.0)
        self.orbit(0.6, run=2.4)
        self.wait(0.5)


# =============================================================== sound (reuses Page 143 sounds)
def build_audio(data):
    total = data["total"]
    buf = np.zeros(int((total + 2.0) * SR))
    log = []

    def add(name, t, s):
        log.append((round(t, 2), name))
        i = int(t * SR)
        if 0 <= i < len(buf):
            j = min(len(buf), i + len(s)); buf[i:j] += s[:j - i]

    def stag(c, fn):
        n, lag, at, d = c["n"], c["lag"], c["at"], max(c["dur"], 0.05)
        tot = 1 + (n - 1) * lag
        for i in range(n):
            fn(i, c["t"] + d * (i * lag + at) / tot)

    for e in data["events"]:
        if "Camera" in e["kinds"] and e["dur"] >= 0.6:
            add("whoosh", e["t"], snd_whoosh(e["dur"]))
        elif "Write" in e["kinds"]:
            add("caption-kazoo", e["t"], snd_caption(e["dur"]))

    for c in data["cues"]:
        nm, t, d = c["name"], c["t"], max(c["dur"], 0.05)
        if nm == "pop":
            add("pop", t, snd_pop(1.0, 0.35))
        elif nm == "say":
            add("say-tok", t, snd_tok(0.13, 900)); add("say-pop", t + 0.05, snd_pop(1.6, 0.12))
        elif nm == "build":
            add("build-whistle", t, snd_whistle(200, 700, d * 0.9, 0.22)); add("build-tok", t + d * 0.9, snd_tok(0.25, 560))
        elif nm == "grow":
            add("grow", t, snd_whistle(220, 600, 0.22, 0.28)); add("squeak", t + 0.1, snd_squeak(1.1, 0.14))
        elif nm == "stagger":
            stag(c, lambda i, ti: add("bloop", ti, snd_bloop(1 + 0.03 * i, 0.3)))
        elif nm == "drop":
            def _d(i, ti):
                add("plink", ti, snd_plink(i, 0.3)); add("squeak", ti + 0.03, snd_squeak(1 + 0.05 * i, 0.12))
            stag(c, _d)
        elif nm == "snap":
            add("snap-whistle", t, snd_whistle(900, 300, min(d, 0.3), 0.2)); add("snap-tok", t + d * 0.9, snd_tok(0.28, 640))
        elif nm == "squash":
            add("squash-whistle", t, snd_whistle(1000, 260, d * 0.9, 0.22)); add("boing", t + d * 0.85, snd_boing(0.25))
        elif nm == "cut":
            add("zip", t, snd_zip(d)); add("cut-tok", t + max(d - 0.06, 0), snd_tok(0.3, 560))
        elif nm == "zap":
            add("zap-xylo", t, snd_xylo(5, 0.28)); add("zap-whistle", t, snd_whistle(500, 1100, 0.2, 0.18))
        elif nm == "ping":
            add("ping", t, snd_ping(c["i"], 0.4 if c["i"] < 3 else 0.3))
        elif nm == "fade":
            add("swish", t, snd_swish(0.45))
        elif nm == "wipe":
            add("wipe-whistle", t, snd_whistle(1500, 350, 0.4, 0.2)); add("swish", t, snd_swish(0.5))
        elif nm == "finale":
            add("TA-DA", t, snd_tada(0.3))

    buf = buf[:int((total + 0.8) * SR)]
    buf *= 0.89 / (np.abs(buf).max() + 1e-9)
    buf = np.tanh(buf * 1.15) / np.tanh(1.15)
    return buf, log


# =============================================================== pipeline
def main():
    os.chdir(ROOT)
    res = os.environ.get("P144_RES", "1080,1080")
    name = "Page144_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page144_timeline.json"
    wav = ROOT / "media/sfx/Page144.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_144.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page144"],
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
    sheet = out_dir / (name + "_sheet.png")          # contact sheet: eyeball text/object overlap
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(final), "-vf",
                    "fps=1/4,scale=360:-1,tile=4x4", "-frames:v", "1", str(sheet)])
    print("=== SFX REPORT ===")
    print(f"timeline {data['total']:.2f}s | {len(data['events'])} plays | {len(data['cues'])} cues | {len(slog)} sounds")
    print("sounds:", dict(Counter(n for _, n in slog)))
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
