#!/usr/bin/env python3
"""Two characters (Tom the fox, Tony the elephant), one voice track each.
  python tools/page_duo.py N        build Page{N}_voice_tom.mp4 from Page{N}_1440x1440_sfx.mp4
  python tools/page_duo.py --smoke  4 s test clip, no network
Narration lines: 'TONY: question | TOM: answer'  (no prefix = TOM)"""
import os, sys, json, math, glob, hashlib, subprocess, pathlib, wave, re
import numpy as np
from PIL import Image
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import tom
import voice_eleven as V

OUT = ROOT / "assets/user-data/output"
CACHE = ROOT / "media/voice/duo_cache"
CACHE.mkdir(parents=True, exist_ok=True)
SR = 24000
DLANG = os.environ.get("DUO_LANG", "en")
SUF = "" if DLANG == "en" else "_" + DLANG
CHARS = json.loads((ROOT / "lib3d/characters.json").read_text())

def ff(*a):
    r = subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", *a], capture_output=True, text=True)
    if r.returncode:
        sys.exit("ffmpeg failed: " + r.stderr[-600:])

def tts(text, who, language=DLANG, expressive=False):
    vp = ROOT / CHARS[who]["voice"]
    if language != "en":
        alt = vp.with_name(vp.stem + "." + language + vp.suffix)
        if alt.exists():
            vp = alt
    ch = json.loads(vp.read_text()) if vp.exists() else CHARS[who]["default_voice"]
    if not ch.get("voice_id"):
        sys.exit(f"no voice set for {who}")
    model = ch.get("model", "eleven_multilingual_v2")
    vs = {"stability": ch.get("stability", 0.45), "similarity_boost": ch.get("similarity", 0.8),
          "style": ch.get("style", 0.25), "use_speaker_boost": True}
    if who == "tony":
        vs["speed"] = 1.05
    if expressive and who == "tom":
        vs.update(stability=0.35, style=0.44, speed=1.06)
    h = hashlib.sha1(json.dumps([ch["voice_id"], model, vs, text, SR], sort_keys=True).encode()).hexdigest()[:16]
    wav = CACHE / f"{h}.wav"
    if not wav.exists():
        mp3 = CACHE / f"{h}.mp3"
        V.synth(text, ch["voice_id"], model, vs, mp3)
        ff("-i", str(mp3), "-ar", str(SR), "-ac", "1", str(wav))
    return wav

def read(p):
    with wave.open(str(p)) as wf:
        return np.frombuffer(wf.readframes(wf.getnframes()), dtype="<i2").astype(np.float32) / 32768

def write(p, x):
    with wave.open(str(p), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SR)
        wf.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())

def speedup(p, f):
    out = CACHE / f"{p.stem}_x{int(f * 100)}.wav"
    if not out.exists():
        ff("-i", str(p), "-filter:a", f"atempo={f:.3f}", "-ar", str(SR), "-ac", "1", str(out))
    return out

def parse(path):
    scenes = []
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        segs = []
        for part in ln.split("|"):
            m = re.match(r"^\s*(TOM|TONY)\s*:\s*(.*)$", part.strip(), re.I)
            who, txt = (m.group(1).lower(), m.group(2).strip()) if m else ("tom", part.strip())
            if txt:
                segs.append((who, txt))
        scenes.append(segs)
    return scenes

def scene_starts(N, n, dur):
    cuts = []
    f = ROOT / f"media/sfx/page_{N}_cuts.json"
    if f.exists():
        try:
            cuts = sorted(float(c["t"]) for c in json.loads(f.read_text()))
        except Exception:
            cuts = []
    if len(cuts) == n - 1:
        print("timing: real cut cues", [0.0] + cuts)
        return [0.0] + cuts
    print(f"timing: no usable cut cues ({len(cuts)} for {n} scenes), even spread")
    return [dur * i / n for i in range(n)]

def ts(t):
    h, m, s = int(t // 3600), int(t % 3600 // 60), t % 60
    return f"{h:02d}:{m:02d}:{int(s):02d},{int((s - int(s)) * 1000):03d}"

def rigs(height):
    out = {}
    for k in ("tom", "tony"):
        c = CHARS[k]
        os.environ["TOM_SVG"] = str(ROOT / "assets/svg" / c["svg"])
        os.environ["TOM_EYES"] = c.get("eyes", "")
        os.environ["TOM_MOUTH"] = c.get("mouth", "")
        os.environ["TOM_BROW"] = c.get("brow", "")
        out[k] = tom.load_rig(height)
    return out

def frame(rig, key, i, a, fps, toff):
    t = i / fps + toff
    os.environ["TOM_FLIP"] = CHARS[key].get("flip", "0")
    body = tom.pose(rig, a, t)
    tw0, th0 = rig["size"]
    br = 0.012 * math.sin(2 * math.pi * 0.5 * t)
    sx, sy = 1 - 0.035 * a - br * 0.5, 1 + 0.07 * a + br
    hop = int(0.05 * th0 * a)
    tilt = 2.5 * math.sin(2 * math.pi * 0.7 * t) * (0.3 + a)
    tw, th = max(1, int(tw0 * sx)), max(1, int(th0 * sy))
    img = body.resize((tw, th), Image.BICUBIC)
    cw, ch = int(tw0 * 1.3), int(th0 * 1.2)
    canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    canvas.alpha_composite(img, ((cw - tw) // 2, max(0, ch - th - 6 - hop)))
    return canvas.rotate(tilt, resample=Image.BICUBIC, center=(cw / 2, ch - 6))

def compose_duo(base, tom_wav, tony_wav, out, height):
    margin = int(os.environ.get("TOM_MARGIN", 24))
    gap = int(os.environ.get("DUO_GAP", 10))
    vv = float(os.environ.get("VOICE_VOL", 0.9))
    tg = vv * float(os.environ.get("TOM_GAIN", 1.7))
    yg = vv * float(os.environ.get("TONY_GAIN", 1.7))
    W, H, fps, dur, _ = tom.probe(base)
    n = int(round(dur * fps))
    R = rigs(height)
    env = {"tom": tom.voice_envelope(str(tom_wav), fps, n), "tony": tom.voice_envelope(str(tony_wav), fps, n)}
    g = lambda e, i: float(e[i]) if i < len(e) else 0.0
    sz = {k: (int(R[k]["size"][0] * 1.3), int(R[k]["size"][1] * 1.2)) for k in R}
    TW = sz["tony"][0] + gap + sz["tom"][0]
    TH = max(sz["tony"][1], sz["tom"][1])
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-i", str(base),
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{TW}x{TH}", "-r", f"{fps}", "-i", "-",
           "-i", str(tom_wav), "-i", str(tony_wav),
           "-filter_complex",
           f"[0:v][1:v]overlay=main_w-overlay_w-{margin}:main_h-overlay_h-{margin // 2}:shortest=1[v];"
           f"[2:a]volume={tg}[a1];[3:a]volume={yg}[a2];"
           f"[a1][a2]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[a]",
           "-map", "[v]", "-map", "[a]", "-t", f"{dur:.3f}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "18", "-c:a", "aac", "-b:a", "192k", str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            canvas = Image.new("RGBA", (TW, TH), (0, 0, 0, 0))
            fy = frame(R["tony"], "tony", i, g(env["tony"], i), fps, 1.3)
            ft = frame(R["tom"], "tom", i, g(env["tom"], i), fps, 0.0)
            canvas.alpha_composite(fy, (0, TH - fy.height))
            canvas.alpha_composite(ft, (sz["tony"][0] + gap, TH - ft.height))
            p.stdin.write(canvas.tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait() != 0:
        sys.exit("ffmpeg failed during compose")
    print("WROTE", out)

def build(N):
    base = OUT / f"Page{N}_1440x1440_sfx.mp4"
    ext = "" if DLANG == "en" else "." + DLANG
    nf = next(iter(glob.glob(str(ROOT / f"scenes/*/page_{N}.narration{ext}.txt"))), None)
    if not base.exists():
        sys.exit(f"missing {base}")
    if not nf:
        sys.exit(f"no narration file for page {N}")
    scenes = parse(pathlib.Path(nf))
    W, H, fps, dur, _ = tom.probe(base)
    starts = scene_starts(N, len(scenes), dur)
    total = int(math.ceil(dur * SR)) + SR
    tracks = {"tom": np.zeros(total, np.float32), "tony": np.zeros(total, np.float32)}
    srt = []
    for i, segs in enumerate(scenes):
        s0 = starts[i]
        s1 = (starts[i + 1] if i + 1 < len(starts) else dur) - 0.3
        files = [(w, t, tts(t, w)) for w, t in segs]
        lens = [len(read(p)) / SR for _, _, p in files]
        gap, lead = 0.2, 0.25
        need = lead + sum(lens) + gap * (len(lens) - 1)
        avail = s1 - s0
        f = 1.0
        if need > avail:
            f = min(1.25, need / avail)
            files = [(w, t, speedup(p, f)) for w, t, p in files]
            lens = [l / f for l in lens]
            need = lead + sum(lens) + gap * (len(lens) - 1)
        flag = "OK" if need <= avail + 0.05 else f"OVER by {need - avail:.1f}s"
        print(f"scene {i + 1}: {sum(len(t.split()) for _, t in segs)} words, {need:.1f}s of {avail:.1f}s, speed x{f:.2f}  {flag}")
        t_cur = s0 + lead
        for (w, t, p), l in zip(files, lens):
            x = read(p)
            a = int(t_cur * SR)
            x = x[: max(0, len(tracks[w]) - a)]
            tracks[w][a:a + len(x)] += x
            srt.append((t_cur, t_cur + l, CHARS[w]["name"], t))
            t_cur += l + gap
    wt = ROOT / f"media/voice/duo_{N}{SUF}_tom.wav"
    wy = ROOT / f"media/voice/duo_{N}{SUF}_tony.wav"
    write(wt, tracks["tom"]); write(wy, tracks["tony"])
    with open(OUT / f"Page{N}_voice{SUF}.srt", "w") as fh:
        for k, (a, b, nm, tx) in enumerate(srt, 1):
            fh.write(f"{k}\n{ts(a)} --> {ts(b)}\n{nm}: {tx}\n\n")
    compose_duo(base, wt, wy, OUT / f"Page{N}_voice_tom{SUF}.mp4", int(os.environ.get("TOM_H", 190)))

def smoke():
    base = pathlib.Path("/tmp/duo_base.mp4")
    ff("-f", "lavfi", "-i", "color=c=0x002A66:s=1240x1080:r=30:d=4", "-pix_fmt", "yuv420p", str(base))
    t = np.arange(SR * 5) / SR
    sig = (0.4 * np.sin(2 * np.pi * 180 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    x = np.zeros(SR * 5, np.float32); y = np.zeros(SR * 5, np.float32)
    x[int(0.3 * SR):int(1.8 * SR)] = sig[:int(1.5 * SR)]
    y[int(2.1 * SR):int(3.6 * SR)] = sig[:int(1.5 * SR)]
    write("/tmp/duo_tom.wav", x); write("/tmp/duo_tony.wav", y)
    compose_duo(base, "/tmp/duo_tom.wav", "/tmp/duo_tony.wav", pathlib.Path("/tmp/duo_smoke.mp4"), 190)
    subprocess.run(["open", "/tmp/duo_smoke.mp4"])

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--smoke":
        smoke()
    elif len(sys.argv) > 1:
        build(sys.argv[1])
    else:
        print(__doc__)
