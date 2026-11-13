"""Page 148 - k-means algorithm, ill-posed clustering, distributed vs one-hot.
Built on page 145 (same helpers, same sound pipeline).
Run:   python3 scenes/chapter05/page_148.py            (1080x1080 -> assets/user-data/output)
Quick: P148_RES=540,540 python3 scenes/chapter05/page_148.py
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

MU = [r"$\mu^{(1)}$", r"$\mu^{(2)}$", r"$\mu^{(3)}$"]


def blob(rng, c, n, s):
    return c + rng.normal(size=(n, 2)) * s


class Page148(Page145):
    def construct(self):
        self.setup_stage("oblique", 0.65)
        self._v = "oblique"; self._sub = None
        self.s_kmeans(); self.wipe()
        self.s_illposed(); self.wipe()
        self.s_distributed(); self.wipe()
        self.wait(0.4)

    # ---- the k-means algorithm
    def s_kmeans(self):
        self.caption("k-means: alternate two steps until convergence")
        self.say(r"Initialize $k$ centroids $\{\mu^{(1)}, \ldots, \mu^{(k)}\}$")
        self.go("oblique")
        self.floor(-0.05)
        rng = np.random.default_rng(148)
        centers = [np.array([-2.7, -1.3]), np.array([2.5, -1.5]), np.array([0.2, 2.5])]
        P = np.vstack([blob(rng, c, 12, 0.5) for c in centers])
        dots = [Dot3D(np.array([x, y, 0.1]), radius=0.1, color=K) for x, y in P]
        run, lag = 1.6, 0.03
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in dots], lag_ratio=lag), run=run, n=len(dots), lag=lag, at=0.6)
        self.keep(*dots)
        C = np.array([[-0.9, -0.3], [0.9, -0.5], [0.0, 1.0]])
        cm = [sphere(0.17, np.array([c[0], c[1], 0.17])) for c in C]
        self.act("grow", *[FadeIn(m) for m in cm], run=0.7)
        self.keep(*cm)
        for it in range(3):
            self.say("Step one: assign each example to the nearest centroid")
            d = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=2)
            a = d.argmin(axis=1)
            lines = [Line(np.array([p[0], p[1], 0.12]), np.array([C[k][0], C[k][1], 0.17]),
                          color=K, stroke_width=2) for p, k in zip(P, a)]
            self.act("cut", *[Create(l) for l in lines], run=0.9)
            self.keep(*lines)
            self.wait(0.3)
            self.say(r"Step two: move each centroid to the mean of its examples")
            newC = np.array([P[a == k].mean(axis=0) if np.any(a == k) else C[k] for k in range(3)])
            self.act("zap", *[cm[k].animate.move_to(np.array([newC[k][0], newC[k][1], 0.17])) for k in range(3)],
                     FadeOut(VGroup(*lines)), run=0.9)
            _forget(self, lines)
            C = newC
        self.say("Repeat until the assignments stop changing")
        self.cue("ping", 0.5, i=0)
        self.play(*[lit(m) for m in cm], run_time=0.6)
        self.label([(MU[k], np.array([C[k][0] + 0.5, C[k][1] + 0.5, 0.3])) for k in range(3)])
        self.orbit(0.4, run=1.8)

    # ---- clustering is ill posed
    def _vehicles(self):
        rng = np.random.default_rng(1481)
        pos = {"rc": (-2.6, -1.7), "rt": (-2.6, 1.7), "gc": (2.6, -1.7), "gt": (2.6, 1.7)}
        G = {}
        for key, c in pos.items():
            pts = blob(rng, np.array(c), 8, 0.35)
            G[key] = [Dot3D(np.array([x, y, 0.1]), radius=0.1, color=K) for x, y in pts]
        return pos, G

    def s_illposed(self):
        self.caption("Clustering is inherently ill posed")
        self.say("Data: red cars, red trucks, gray cars, gray trucks")
        self.go("oblique")
        self.floor(-0.05)
        pos, G = self._vehicles()
        alld = [d for g in G.values() for d in g]
        run, lag = 1.6, 0.03
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in alld], lag_ratio=lag), run=run, n=len(alld), lag=lag, at=0.6)
        self.keep(*alld)
        self.label([("red cars", np.array([-2.6, -2.9, 0])), ("red trucks", np.array([-2.6, 2.9, 0])),
                    ("gray cars", np.array([2.6, -2.9, 0])), ("gray trucks", np.array([2.6, 2.9, 0]))])

        def paint(keys):
            self.cue("ping", 0.5, i=0)
            self.play(*[d.animate.set_color(ACCENT) for k in keys for d in G[k]], run_time=0.5)

        def clear(keys):
            self.play(*[d.animate.set_color(K) for k in keys for d in G[k]], run_time=0.4)

        self.say("One algorithm finds a cluster of cars and a cluster of trucks")
        paint(["rc", "gc"]); self.wait(0.6); clear(["rc", "gc"])
        paint(["rt", "gt"]); self.wait(0.6); clear(["rt", "gt"])
        self.say("Another finds red vehicles and gray vehicles")
        paint(["rc", "rt"]); self.wait(0.6); clear(["rc", "rt"])
        paint(["gc", "gt"]); self.wait(0.6); clear(["gc", "gt"])
        self.say("A third finds four clusters: both attributes are captured")
        for k in ("rc", "rt", "gc", "gt"):
            paint([k]); self.wait(0.3); clear([k])
        self.say("But similarity is lost: red cars differ from gray cars and from gray trucks")
        a = np.array([-2.6, -1.7, 0.15])
        ls = [Line(a, np.array([2.6, -1.7, 0.15]), color=ACCENT, stroke_width=5),
              Line(a, np.array([2.6, 1.7, 0.15]), color=ACCENT, stroke_width=5)]
        self.act("cut", *[Create(l) for l in ls], run=1.0)
        self.keep(*ls)
        self.orbit(0.4, run=1.8)

    # ---- distributed representation
    def s_distributed(self):
        self.caption("A distributed representation uses many attributes")
        self.say("Two attributes per vehicle: color, and car or truck")
        self.go("oblique")
        self.floor(-0.05)
        pos, G = self._vehicles()
        alld = [d for g in G.values() for d in g]
        run, lag = 1.4, 0.03
        self.act("stagger", LaggedStart(*[FadeIn(d) for d in alld], lag_ratio=lag), run=run, n=len(alld), lag=lag, at=0.6)
        self.keep(*alld)
        ax = [arrow3d(np.array([-4.2, -3.6, 0.03]), np.array([4.2, -3.6, 0.03]), color=K),
              arrow3d(np.array([-4.2, -3.6, 0.03]), np.array([-4.2, 3.6, 0.03]), color=K)]
        self.act("pop", *[FadeIn(a) for a in ax], run=0.6)
        self.keep(*ax)
        self.label([("color", np.array([4.3, -3.9, 0])), ("car or truck", np.array([-4.6, 3.9, 0]))])
        self.say("Shared color: red cars and red trucks are alike in one attribute")
        self.cue("ping", 0.5, i=2)
        self.play(*[d.animate.set_color(ACCENT) for k in ("rc", "rt") for d in G[k]], run_time=0.5)
        self.wait(0.7)
        self.play(*[d.animate.set_color(K) for k in ("rc", "rt") for d in G[k]], run_time=0.4)
        self.say("Shared type: red cars and gray cars are alike in the other")
        self.cue("ping", 0.5, i=0)
        self.play(*[d.animate.set_color(ACCENT) for k in ("rc", "gc") for d in G[k]], run_time=0.5)
        self.wait(0.7)
        self.say("Comparing many attributes measures similarity in a fine-grained way")
        self.cue("finale", 1.0)
        self.wait(1.2)
        self.orbit(0.4, run=1.8)


# ================================================================ pipeline
def main():
    os.chdir(ROOT)
    res = os.environ.get("P148_RES", "1080,1080")
    name = "Page148_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page148_timeline.json"
    wav = ROOT / "media/sfx/Page148.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_148.log")
    print(f"rendering {res} @30fps ...  (progress: bash lib3d/watch_progress.sh /tmp/page_148.log)")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page148"],
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
