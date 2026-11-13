"""Page 143 - Figure 5.7: decision tree (top) and the space it divides (bottom).
Self-contained: scene + funny sound design + render + mux.
Run:   python3 scenes/chapter05/page_143.py            (1080x1080, saves to assets/user-data/output)
Quick: P143_RES=540,540 python3 scenes/chapter05/page_143.py
"""
import json
import os
import pathlib
import subprocess
import sys
import wave
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from manim import *
from lib3d import *
from lib3d.ml.basics_fit import _stage_floor

# =============================================================== data
TREE = {"": "I", "0": "I", "1": "I", "00": "L", "01": "I", "10": "L", "11": "I",
        "010": "L", "011": "L", "110": "L", "111": "I", "1110": "L", "1111": "L"}


def _inorder(i=""):
    r = []
    if i + "0" in TREE:
        r += _inorder(i + "0")
    r.append(i)
    if i + "1" in TREE:
        r += _inorder(i + "1")
    return r


ORDER = _inorder()
P = {i: np.array([(ORDER.index(i) - 6) * 0.5, len(i) * 0.3 - 0.6, 2.3 - len(i) * 1.15]) for i in TREE}

# bottom figure: (node id, cut start xy, cut end xy, node xy)
CUTS = [("", (-3, 0.3), (3, 0.3), (-0.8, 0.3)),
        ("0", (0.9, 0.3), (0.9, 3), (0.9, 0.9)),
        ("01", (0.9, 1.7), (3, 1.7), (1.9, 1.7)),
        ("1", (-1.6, -3), (-1.6, 0.3), (-1.6, -0.5)),
        ("11", (-1.6, -1.2), (3, -1.2), (1.4, -1.2)),
        ("111", (0.3, -3), (0.3, -1.2), (0.3, -2.1))]
# leaf id -> (x0, x1, y0, y1, slab height)
REGIONS = {"00": (-3, 0.9, 0.3, 3, 0.5), "010": (0.9, 3, 1.7, 3, 1.3), "011": (0.9, 3, 0.3, 1.7, 0.8),
           "10": (-3, -1.6, -3, 0.3, 1.0), "110": (-1.6, 3, -1.2, 0.3, 0.6),
           "1110": (-1.6, 0.3, -3, -1.2, 1.5), "1111": (0.3, 3, -3, -1.2, 2.0)}
LEAF_ORDER = ["00", "010", "011", "10", "110", "1110", "1111"]


def _forget(scene, mobs):
    it = scene._items()
    for m in mobs:
        try:
            it.remove(m)
        except ValueError:
            pass


# =============================================================== scene
class Page143(Stage3D):
    # ---- cue + timeline recording (this is what keeps sound in sync)
    def _now(self):
        return float(self.time)

    def cue(self, name, dur=0.0, **kw):
        if not hasattr(self, "_cues"):
            self._cues = []
        d = {"t": self._now(), "name": name, "dur": float(dur)}
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
        path = os.environ.get("SFX_TIMELINE", str(ROOT / "media/sfx/Page143_timeline.json"))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump({"total": self._now(), "events": getattr(self, "_events", []),
                       "cues": getattr(self, "_cues", [])}, f)

    def wipe(self):
        self.cue("wipe", 0.7)
        self.clear_scene()

    # ---- animation
    def construct(self):
        self.setup_stage("slant", 0.7)
        self.tree_scene()
        self.wipe()
        self.partition_scene()
        self.wipe()
        self.wait(0.4)

    def tree_scene(self):
        self.caption("Decision tree: left = 0, right = 1")
        self.cue("pop", 0.5)
        _stage_floor(self, ORIGIN, z=-3.2)
        mob = {i: (sphere(0.2, P[i]) if TREE[i] == "I" else box((0.34, 0.34, 0.34), P[i])) for i in TREE}
        edge = {i: Line(P[i[:-1]], P[i], color=K, stroke_width=3) for i in TREE if i}
        self.keep(*mob.values(), *edge.values())

        self.cue("node", 0.5)
        self.play(FadeIn(mob[""]), run_time=0.5)
        for lv in range(1, 5):
            ids = [i for i in ORDER if len(i) == lv]
            lag, run = 0.3, 0.6 + 0.35 * len(ids)
            self.cue("spawn", run, lag=lag, inner=1.5, at=1.0, edges=True,
                     kinds="".join(TREE[i] for i in ids))
            self.play(LaggedStart(*[AnimationGroup(Create(edge[i]), FadeIn(mob[i]), lag_ratio=0.5)
                                    for i in ids], lag_ratio=lag), run_time=run)
            self.cue("labels", 0.4, n=len(ids))
            tags(self, [(i, P[i] + np.array([0, 0, 0.45])) for i in ids], 24)

        self.cue("labels", 0.4, n=2)
        tags(self, [("circle = internal node", np.array([-2.2, 0, -3.7])),
                    ("square = leaf", np.array([2.2, 0, -3.7]))], 24)
        self.orbit(0.5, run=2.0)

        self.caption("An example follows one path to one leaf")

        def route(path):
            done = []
            self.cue("ping", 0.3, i=0)
            self.play(lit(mob[path[0]]), run_time=0.3)
            done.append(path[0])
            for k in range(1, len(path)):
                self.cue("hop", 0.5, i=k)
                flashes(self, [(P[path[k - 1]], P[path[k]])], run=0.5)
                if TREE[path[k]] == "L":
                    self.cue("land", 0.5)
                else:
                    self.cue("ping", 0.3, i=k)
                self.play(lit(mob[path[k]]), run_time=0.4)
                done.append(path[k])
            self.wait(0.4)
            self.cue("reset", 0.5)
            self.play(*[unlit(mob[i]) for i in done], run_time=0.5)

        route(["", "0", "01", "011"])
        route(["", "1", "11", "111", "1111"])

    def partition_scene(self):
        self.caption("The tree divides the plane into regions")
        self.view("oblique", 1.4)
        self.cue("pop", 0.5)
        _stage_floor(self, ORIGIN, z=-0.02)
        c = [np.array(p + (0.03,)) for p in ((-3, -3), (3, -3), (3, 3), (-3, 3))]
        frame = [segment(c[k], c[(k + 1) % 4], width=6) for k in range(4)]
        self.cue("cut", 0.9)
        self.play(*[Create(s) for s in frame], run_time=0.9)
        self.keep(*frame)

        lbl_int, int_nodes = [], []
        for cid, a, b, n in CUTS:
            ln = segment(np.array([*a, 0.05]), np.array([*b, 0.05]), width=6, color=ACCENT)
            nd = sphere(0.2, np.array([*n, 0.2]))
            self.cue("cut", 0.7)
            self.play(Create(ln), run_time=0.7)
            self.cue("node", 0.4)
            self.play(FadeIn(nd), ln.animate.set_color(K), run_time=0.4)
            self.keep(ln, nd)
            int_nodes.append(nd)
            if cid:
                self.cue("labels", 0.4, n=1)
                lbl_int += tags(self, [(cid, np.array([*n, 0.7]))], 24)

        # leaves land in the centre of their regions
        leaves, lbl_leaf = [], []
        for lid in LEAF_ORDER:
            x0, x1, y0, y1, _ = REGIONS[lid]
            leaves.append(box((0.34, 0.34, 0.34), np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.18])))
        lag, run = 0.25, 2.0
        self.cue("spawn", run, lag=lag, inner=1.0, at=0.4, edges=False, kinds="L" * len(leaves))
        self.play(LaggedStart(*[FadeIn(b) for b in leaves], lag_ratio=lag), run_time=run)
        self.keep(*leaves)
        self.cue("labels", 0.4, n=len(leaves))
        for lid in LEAF_ORDER:
            x0, x1, y0, y1, _ = REGIONS[lid]
            lbl_leaf += tags(self, [(lid, np.array([(x0 + x1) / 2, (y0 + y1) / 2, 0.65]))], 24)
        self.orbit(0.5, run=2.0)

        # piecewise constant: one slab per leaf
        self.caption("Piecewise constant: one piece per leaf")
        gone = int_nodes + leaves + lbl_int + lbl_leaf
        self.cue("fade", 0.5)
        self.play(*[FadeOut(m) for m in gone], run_time=0.5)
        _forget(self, gone)
        slabs = []
        for lid in LEAF_ORDER:
            x0, x1, y0, y1, h = REGIONS[lid]
            slabs.append(box((x1 - x0 - 0.12, y1 - y0 - 0.12, h),
                             np.array([(x0 + x1) / 2, (y0 + y1) / 2, h / 2])))
        lag, run = 0.25, 2.0
        self.cue("rise", run, lag=lag, inner=1.0, at=0.3, n=len(slabs))
        self.play(LaggedStart(*[FadeIn(s, shift=OUT * 0.8) for s in slabs], lag_ratio=lag), run_time=run)
        self.keep(*slabs)
        self.cue("labels", 0.4, n=len(slabs))
        tags(self, [(lid, np.array([(REGIONS[lid][0] + REGIONS[lid][1]) / 2,
                                    (REGIONS[lid][2] + REGIONS[lid][3]) / 2, REGIONS[lid][4] + 0.5]))
                    for lid in LEAF_ORDER], 24)
        self.orbit(0.6, run=2.4)

        # every leaf needs at least one training example
        self.caption("Each leaf needs at least one training example")
        dots = []
        for lid in LEAF_ORDER:
            x0, x1, y0, y1, h = REGIONS[lid]
            dots.append(Dot3D(np.array([(x0 + x1) / 2, (y0 + y1) / 2, h + 0.12]), radius=0.1, color=ACCENT))
        lag, run = 0.3, 2.0
        self.cue("drop", run, lag=lag, inner=1.0, at=0.7, n=len(dots))
        self.play(LaggedStart(*[FadeIn(d, shift=IN * 1.4) for d in dots], lag_ratio=lag), run_time=run)
        self.keep(*dots)
        self.cue("labels", 0.4, n=1)
        tags(self, [("7 leaves = 7 examples: at most 7 local maxima", np.array([0, -3.7, 0.2]))], 24)
        self.cue("finale", 1.0)
        self.play(lit(slabs[-1]), run_time=0.8)
        self.orbit(0.6, run=2.4)
        self.wait(0.5)


# =============================================================== sound design
SR = 44100
_rng = np.random.default_rng(143)
_cap = [0]


def _T(d): return np.arange(max(int(d * SR), 2)) / SR
def _env(t, tau): return np.exp(-t / tau)
def _noise(d): return _rng.uniform(-1, 1, max(int(d * SR), 8))
def _lp(x, k): return np.convolve(x, np.ones(k) / k, "same")
def _hp(x, k=8): return x - _lp(x, k)
def _ph(f): return 2 * np.pi * np.cumsum(f) / SR
def _hump(n, p=1.0): return np.sin(np.pi * np.linspace(0, 1, n)) ** p
def _sw(f0, f1, d):
    t = _T(d)
    return np.sin(_ph(f0 * (f1 / f0) ** (t / d)))


def _atk(n, ms=2):
    a = min(int(SR * ms / 1000), n)
    e = np.ones(n)
    if a > 0:
        e[:a] = np.linspace(0, 1, a)
    return e


def snd_whistle(f0, f1, d, v=0.3):          # slide whistle
    t = _T(d)
    f = np.linspace(f0, f1, len(t)) * (1 + 0.015 * np.sin(2 * np.pi * 7 * t))
    s = np.sin(_ph(f)) + 0.12 * np.sin(2 * _ph(f))
    return v * s * _hump(len(t), 0.5) * _atk(len(t), 4)


def snd_boing(v=0.3, d=0.28):
    t = _T(d)
    f = 240 + 360 * np.sin(np.pi * t / d) ** 0.7
    return v * np.sin(_ph(f)) * (1 + 0.3 * np.sin(2 * np.pi * 28 * t)) * _hump(len(t), 0.6)


def snd_bloop(p=1.0, v=0.35):               # cartoon water-drop
    d = 0.16; t = _T(d)
    f = (260 + 700 * np.sin(np.pi * t / d) ** 1.5) * p
    return v * np.sin(_ph(f)) * _env(t, 0.07) * _atk(len(t))


def snd_squish(v=0.35):                     # leaf lands: soft plop
    d = 0.16; t = _T(d)
    s = np.sin(_ph(180 * (90 / 180) ** (t / d))) * _env(t, 0.05)
    n = _lp(_noise(d), 10)[:len(t)] * _env(t, 0.03)
    return v * (s + 0.5 * n) * _atk(len(t), 1)


def snd_squeak(p=1.0, v=0.2):               # rubber duck
    d = 0.09; t = _T(d)
    return v * np.sin(_ph((1900 + 700 * np.sin(np.pi * t / d)) * p)) * _hump(len(t), 0.5)


def snd_tok(v=0.3, f=620):                  # wood block
    d = 0.09; t = _T(d)
    return v * (np.sin(_ph(f * 0.55 ** (t / d))) * _env(t, 0.02)
                + 0.3 * _hp(_noise(d), 4)[:len(t)] * _env(t, 0.004))


def snd_zip(d, v=0.28):                     # zipper / ratchet cut
    d = min(d, 1.2); t = _T(d)
    gate = 0.5 + 0.5 * np.sign(np.sin(_ph(22 + 50 * (t / d))))
    x = _hp(_noise(d), 4)[:len(t)]
    return v * x * gate * _hump(len(t), 0.4) + 0.04 * np.sin(_ph(300 + 900 * t / d)) * _hump(len(t))


def snd_xylo(semi, v=0.3, base=392):
    f = base * 2 ** (semi / 12); t = _T(0.2)
    s = (np.sin(2 * np.pi * f * t) + 0.45 * np.sin(6 * np.pi * f * t) * _env(t, 0.02)
         + 0.2 * np.sin(4 * np.pi * f * t) * _env(t, 0.05))
    return v * s * _env(t, 0.06) * _atk(len(t), 1)


def snd_kazoo(f, d=0.18, v=0.2):
    t = _T(d)
    ph = _ph(f * (1 + 0.012 * np.sin(2 * np.pi * 6 * t)))
    s = sum(np.sin(k * ph) / k for k in range(1, 9)) + 0.3 * _lp(_noise(d), 3)[:len(t)]
    env = np.minimum(1, t / 0.01) * np.exp(-np.maximum(t - d * 0.6, 0) / 0.04)
    return v * 0.6 * s * env


def snd_ping(i, v=0.4):
    f = 660 * 2 ** (([0, 2, 4, 7, 9][i % 5] + 12 * (i // 5)) / 12); t = _T(0.5)
    s = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    return v * s * _env(t, 0.13) * _atk(len(t))


def snd_pop(p=1.0, v=0.35):
    d = 0.14; t = _T(d)
    return v * _sw(720 * p, 260 * p, d) * _env(t, 0.04) * _atk(len(t))


def snd_plink(i, v=0.3):
    d = 0.16; t = _T(d)
    return v * _sw(1500 * 1.05 ** i, 900 * 1.05 ** i, d) * _env(t, 0.05) * _atk(len(t), 1)


def snd_whoosh(d):
    d = min(d, 2.5); n = int(d * SR)
    x = _lp(_noise(d), 24)[:n]; x = x / (np.abs(x).max() + 1e-9)
    return 0.5 * (0.8 * x + 0.25 * _sw(120, 420, d)[:n]) * _hump(n, 1.0) ** 2


def snd_swish(d=0.4, v=0.18):
    t = _T(d)
    x = _lp(_noise(d), 16)[:len(t)]; x = x / (np.abs(x).max() + 1e-9)
    return v * x * _env(t, d / 3) * _atk(len(t), 8)


def snd_cymbal(v=0.2):
    t = _T(1.0)
    return v * _hp(_noise(1.0), 3)[:len(t)] * _env(t, 0.3)


_PHR = [[0, 2, 4, 5, 7, 9, 11, 12], [12, 9, 7, 4, 5, 2, 0, -5],
        [0, 7, 4, 7, 2, 7, 0, 12], [0, 4, 0, 5, 0, 7, 9, 12]]


def snd_caption(d):                         # kazoo tiptoe + boing
    dd = min(max(d, 0.5), 1.2)
    notes = _PHR[_cap[0] % len(_PHR)]; _cap[0] += 1
    n = len(notes); step = dd / (n + 1)
    out = np.zeros(int((dd + 0.5) * SR))
    for i, s in enumerate(notes):
        k = snd_kazoo(330 * 2 ** (s / 12), min(step * 1.6, 0.2), 0.17)
        p = int(i * step * SR); out[p:p + len(k)] += k[:len(out) - p]
    b = snd_boing(0.22); p = int(n * step * SR)
    out[p:p + len(b)] += b[:max(0, len(out) - p)]
    return out


def snd_tada(v=0.3):
    out = np.zeros(int(1.4 * SR))
    for st, semi, d in ((0.0, 0, 0.14), (0.14, 4, 0.14), (0.28, 7, 0.55)):
        k = snd_kazoo(440 * 2 ** (semi / 12), d, 0.28 * v / 0.3)
        p = int(st * SR); out[p:p + len(k)] += k[:len(out) - p]
    c = snd_cymbal(0.18); p = int(0.28 * SR); out[p:p + len(c)] += c[:len(out) - p]
    return out


def snd_land():
    out = np.zeros(int(0.7 * SR))
    for st, s in ((0.0, snd_xylo(7, 0.28)), (0.07, snd_xylo(12, 0.28)),
                  (0.16, snd_boing(0.3)), (0.3, snd_squeak(1.2, 0.2))):
        p = int(st * SR); out[p:p + len(s)] += s[:len(out) - p]
    return out


def build_audio(data):
    total = data["total"]
    buf = np.zeros(int((total + 2.0) * SR))
    log = []

    def add(name, t, s):
        log.append((round(t, 2), name))
        i = int(t * SR)
        if 0 <= i < len(buf):
            j = min(len(buf), i + len(s)); buf[i:j] += s[:j - i]

    for e in data["events"]:
        if "Camera" in e["kinds"] and e["dur"] >= 0.6:
            add("whoosh", e["t"], snd_whoosh(e["dur"]))
        elif "Write" in e["kinds"]:
            add("caption-kazoo", e["t"], snd_caption(e["dur"]))

    for c in data["cues"]:
        nm, t, d = c["name"], c["t"], max(c["dur"], 0.05)
        if nm == "pop":
            add("pop", t, snd_pop(1.0, 0.35))
        elif nm == "node":
            add("bloop", t + 0.1 * d, snd_bloop(1.0, 0.4)); add("squeak", t + 0.2 * d, snd_squeak(1.0, 0.15))
        elif nm == "spawn":
            kinds, lag, inner, at = c["kinds"], c["lag"], c["inner"], c["at"]
            n = len(kinds); u = d / (inner * (1 + (n - 1) * lag))
            for i, k in enumerate(kinds):
                t0 = t + i * lag * inner * u
                if c.get("edges"):
                    add("edge-tok", t0, snd_tok(0.22, 700))
                tn = t0 + at * u
                if k == "I":
                    add("bloop", tn, snd_bloop(1 + 0.04 * i, 0.38))
                else:
                    add("squish", tn, snd_squish(0.4)); add("squeak", tn + 0.05, snd_squeak(1 + 0.06 * i, 0.18))
        elif nm == "labels":
            for i in range(min(c["n"], 12)):
                add("label-tick", t + i * d / max(c["n"], 1), snd_tok(0.12, 900 + 40 * i))
        elif nm == "hop":
            add("hop-xylo", t, snd_xylo(2 * c["i"], 0.3)); add("hop-whistle", t, snd_whistle(500, 900, 0.14, 0.18))
        elif nm == "ping":
            add("ping", t, snd_ping(c["i"], 0.4 if c["i"] < 3 else 0.3))
        elif nm == "land":
            add("land", t, snd_land())
        elif nm == "reset":
            add("reset-whistle", t, snd_whistle(900, 400, 0.25, 0.16))
        elif nm == "cut":
            add("zip", t, snd_zip(d)); add("cut-tok", t + max(d - 0.06, 0), snd_tok(0.3, 560))
        elif nm == "fade":
            add("swish", t, snd_swish(0.45))
        elif nm in ("rise", "drop"):
            n, lag, inner, at = c["n"], c["lag"], c["inner"], c["at"]
            u = d / (inner * (1 + (n - 1) * lag))
            for i in range(n):
                ti = t + i * lag * inner * u + at * u
                if nm == "rise":
                    add("sproing", ti, snd_whistle(180 * 1.07 ** i, 520 * 1.07 ** i, 0.24, 0.22))
                else:
                    add("plink", ti, snd_plink(i, 0.3)); add("squeak", ti + 0.03, snd_squeak(1 + 0.05 * i, 0.12))
        elif nm == "finale":
            add("TA-DA", t, snd_tada(0.3))
        elif nm == "wipe":
            add("wipe-whistle", t, snd_whistle(1500, 350, 0.4, 0.2)); add("swish", t, snd_swish(0.5))

    buf = buf[:int((total + 0.8) * SR)]
    buf *= 0.89 / (np.abs(buf).max() + 1e-9)
    buf = np.tanh(buf * 1.15) / np.tanh(1.15)
    return buf, log


def write_wav(buf, path):
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.column_stack([pcm, pcm]).tobytes())


# =============================================================== pipeline
def main():
    os.chdir(ROOT)
    res = os.environ.get("P143_RES", "1080,1080")
    name = "Page143_" + res.replace(",", "x")
    tl = ROOT / "media/sfx/Page143_timeline.json"
    wav = ROOT / "media/sfx/Page143.wav"
    out_dir = ROOT / "assets/user-data/output"
    for p in (tl.parent, out_dir):
        p.mkdir(parents=True, exist_ok=True)
    if tl.exists():
        tl.unlink()
    env = dict(os.environ, SFX_TIMELINE=str(tl), PYTHONPATH=str(ROOT))
    log = pathlib.Path("/tmp/page_143.log")
    print(f"rendering {res} @30fps ...")
    with open(log, "w") as lf:
        rc = subprocess.run([sys.executable, "-m", "manim", "-r", res, "--fps", "30", "-o", name,
                             str(pathlib.Path(__file__).resolve()), "Page143"],
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
    print("=== SFX REPORT ===")
    print(f"timeline {data['total']:.2f}s | {len(data['events'])} plays | {len(data['cues'])} cues | {len(slog)} sounds")
    print("sounds:", dict(Counter(n for _, n in slog)))
    print("first 8:", slog[:8]); print("last 4:", slog[-4:])
    try:
        vd = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                      "-of", "csv=p=0", str(vid)], text=True).strip()
        print(f"video duration {vd}s vs timeline {data['total']:.2f}s (should match within ~0.1s)")
    except Exception:
        pass
    print("SAVED:", final)
    subprocess.run(["open", str(final)])


if __name__ == "__main__":
    main()
