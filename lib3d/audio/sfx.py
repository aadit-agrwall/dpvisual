"""timeline.json -> SFX wav.  Usage: python3 lib3d/audio/sfx.py timeline.json out.wav
Pure numpy. No music, only effects placed on the real animation times.
Text writing = funny Tom-and-Jerry style tiptoe (xylophone plucks + boing)."""
import json
import sys
import wave
from collections import Counter

import numpy as np

SR = 44100
rng = np.random.default_rng(7)


def T(d): return np.arange(int(d * SR)) / SR
def expd(t, tau): return np.exp(-t / tau)
def noise(d): return rng.uniform(-1, 1, max(int(d * SR), 8))
def lp(x, k): return np.convolve(x, np.ones(k) / k, "same")
def hp(x, k=8): return x - lp(x, k)


def atk(n, ms=2):
    a = min(int(SR * ms / 1000), n)
    e = np.ones(n)
    if a > 0:
        e[:a] = np.linspace(0, 1, a)
    return e


def sweep(f0, f1, d):
    t = T(d)
    f = f0 * (f1 / f0) ** (t / d)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def hump(n, p=1.0): return np.sin(np.pi * np.linspace(0, 1, n)) ** p


# ---------------------------------------------------------------- sounds
def pop(p=1.0, v=0.4):
    d = 0.14; t = T(d)
    return v * sweep(720 * p, 260 * p, d) * expd(t, 0.04) * atk(len(t))


def ping(i, v=0.42):
    f = 660 * 2 ** (([0, 2, 4, 7, 9][i % 5] + 12 * (i // 5)) / 12)
    t = T(0.5)
    s = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(4 * np.pi * f * t) + 0.12 * np.sin(6 * np.pi * f * t)
    return v * s * expd(t, 0.13) * atk(len(t))


def untick(v=0.25):
    d = 0.12; t = T(d)
    return v * sweep(480, 300, d) * expd(t, 0.04) * atk(len(t))


def tick(v=0.22, f=1500):
    t = T(0.03)
    return v * (np.sin(2 * np.pi * f * t) + 0.5 * noise(0.03)[:len(t)]) * expd(t, 0.006)


# --- funny "chase" typing: sneaky xylophone/pizzicato run, ends with a boing
_PHRASES = [
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 7],        # tiptoe up the scale
    [0, 4, 2, 5, 4, 7, 5, 9, 7, 12],       # sneaky zig-zag
    [12, 11, 10, 9, 8, 7, 6, 5, 7, 4],     # tiptoe down
    [0, 2, 0, 3, 0, 4, 0, 5, 7, 12],       # peek-a-boo
]
_ph = [0]


def pluck(semi, v=0.3):
    f = 392 * 2 ** (semi / 12)
    t = T(0.17)
    s = (np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 3 * f * t) * expd(t, 0.02)
         + 0.25 * np.sin(2 * np.pi * 2 * f * t) * expd(t, 0.05))
    return v * s * expd(t, 0.055) * atk(len(t), 1)


def boing(v=0.3):
    d = 0.2; t = T(d)
    f = 260 + 380 * np.sin(np.pi * t / d) ** 0.7
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * hump(len(t), 0.6) * (1 + 0.25 * np.sin(2 * np.pi * 30 * t))
    return v * s


def typing(d):
    dd = min(d, 1.4)
    notes = _PHRASES[_ph[0] % len(_PHRASES)]; _ph[0] += 1
    n = int(np.clip(dd / 0.075, 6, len(notes)))
    step = dd / (n + 1)
    out = np.zeros(int((dd + 0.45) * SR))
    for i in range(n):
        s = pluck(notes[i], 0.3 if i % 2 == 0 else 0.22)
        p = int(i * step * SR)
        out[p:p + len(s)] += s
    b = boing(0.28)
    p = int(n * step * SR)
    out[p:p + len(b)] += b[:len(out) - p]
    return out


def draw(d):
    d = min(d, 1.6); n = int(d * SR); t = T(d)
    x = hp(noise(d), 5)[:n] * (0.6 + 0.4 * np.sin(2 * np.pi * 14 * t))
    return 0.22 * x * hump(n, 0.7) + 0.05 * sweep(260, 520, d) * hump(n)


def whoosh(d):
    d = min(d, 2.5); n = int(d * SR)
    x = lp(noise(d), 24)[:n]; x = x / (np.abs(x).max() + 1e-9)
    env = hump(n, 1.0) ** 2
    return 0.5 * (0.8 * x + 0.25 * sweep(120, 420, d)) * env


def fade_out(d):
    d = min(d, 0.5); t = T(d)
    x = lp(noise(d), 16)[:len(t)]; x = x / (np.abs(x).max() + 1e-9)
    return 0.2 * x * expd(t, d / 3) * atk(len(t), 8)


def zap():
    d = 0.16; t = T(d); a = sweep(300, 2600, d)
    return 0.3 * (0.6 * a + 0.4 * np.sign(a) * 0.4) * expd(t, 0.06) * atk(len(t), 1)


def grow():
    d = 0.2
    return 0.35 * sweep(220, 540, d) * hump(int(d * SR))


def bonk():
    out = np.zeros(int(0.8 * SR))
    for start, f0, f1, dur in ((0.0, 260, 150, 0.3), (0.28, 200, 100, 0.45)):
        t = T(dur)
        s = (sweep(f0, f1, dur) + 0.4 * sweep(f0 * 2, f1 * 2, dur)) * expd(t, 0.14) * atk(len(t))
        i = int(start * SR); out[i:i + len(s)] += 0.5 * s
    return out


def chime():
    out = np.zeros(int(0.9 * SR))
    for start, f in ((0.0, 784), (0.12, 1175)):
        t = T(0.7)
        s = (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t)) * expd(t, 0.25) * atk(len(t))
        i = int(start * SR); out[i:i + len(s)] += 0.4 * s
    return out


# ----------------------------------------------------------------- mixer
def build(data):
    total = data["total"]
    buf = np.zeros(int((total + 1.5) * SR))
    log = []

    def add(name, t, s):
        log.append((round(t, 2), name))
        i = int(t * SR)
        if 0 <= i < len(buf):
            j = min(len(buf), i + len(s))
            buf[i:j] += s[:j - i]

    for e in data["events"]:
        t, dur, kinds = e["t"], max(e["dur"], 0.05), e["kinds"]
        n_pop = n_lit = n_zap = 0
        did = set()
        for k in kinds:
            c = k["cls"]
            if c == "Camera":
                if dur >= 0.6:
                    add("whoosh", t, whoosh(dur))
            elif c == "Write":
                add("chase-typing", t, typing(dur))
            elif c == "FadeIn":
                txt = k.get("text", "")
                low = txt.lower()
                if txt:
                    if n_pop < 8:
                        add("label-pop", t + 0.03 * n_pop, pop(1.5, 0.28)); n_pop += 1
                elif k.get("n", 1) > 4:
                    m = min(k["n"], 14)
                    for i in range(m):
                        add("shimmer", t + i * min(dur, 1.2) / m, tick(0.16, rng.uniform(1800, 3200)))
                else:
                    add("pop", t, pop(1.0, 0.4))
                if "wrong" in low:
                    add("BONK-wrong", t + 0.15, bonk())
                if "correct" in low:
                    add("CHIME-correct", t + 0.15, chime())
            elif c == "LaggedStart":
                m = min(k.get("n_sub", 4), 14)
                for i in range(m):
                    add("stagger-pop", t + i * dur * 0.8 / m, pop(0.8 * 1.04 ** i, 0.3))
            elif c == "Create":
                if "draw" not in did:
                    add("draw", t, draw(dur)); did.add("draw")
            elif c == "ShowPassingFlash":
                if n_zap < 4:
                    add("zap", t + 0.04 * n_zap, zap()); n_zap += 1
            elif c == "FadeOut":
                if "fo" not in did:
                    add("fade-out", t, fade_out(dur)); did.add("fo")
            elif c == "GrowFromCenter":
                add("grow", t, grow())
            elif c == "Method":
                st = k.get("state")
                if st == "lit":
                    if n_lit < 6:
                        add("ping", t + 0.05 * n_lit, ping(n_lit, 0.42 if n_lit < 3 else 0.28)); n_lit += 1
                elif st == "unlit":
                    if "un" not in did:
                        add("unlit", t, untick()); did.add("un")
                elif "mv" not in did:
                    add("move-tick", t, tick(0.15, 1100)); did.add("mv")
            else:
                if "ot" not in did:
                    add("tick", t, tick(0.12)); did.add("ot")

    buf = buf[:int((total + 0.5) * SR)]
    buf *= 0.89 / (np.abs(buf).max() + 1e-9)
    buf = np.tanh(buf * 1.15) / np.tanh(1.15)
    return buf, log


def main():
    src, dst = sys.argv[1], sys.argv[2]
    data = json.load(open(src))
    buf, log = build(data)
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(dst, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(np.column_stack([pcm, pcm]).tobytes())
    print("=== SFX REPORT ===")
    print(f"timeline total {data['total']:.2f}s, {len(data['events'])} plays, {len(log)} sounds -> {dst}")
    print("sounds:", dict(Counter(n for _, n in log)))
    print("first 12:", log[:12])
    print("last 5:", log[-5:])


if __name__ == "__main__":
    main()
