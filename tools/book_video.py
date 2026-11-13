import os, pathlib, subprocess, wave
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGES = [int(x) for x in os.environ.get("BV_PAGES", "151,152,153,154,155").split(",")]
OUT = ROOT / "assets/user-data/output"
WORK = ROOT / "media/book_video"
BOOK = ROOT / "assets/bookpage"
WL, WR, H, FPS, TRANS = 1120, 1440, 1440, 30, 1.3
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-r", "30",
       "-video_track_timescale", "15360", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2"]
WORK.mkdir(parents=True, exist_ok=True)


def run(cmd):
    subprocess.run([str(c) for c in cmd], check=True)


def left_panel(n):
    im = Image.open(BOOK / f"page_{n}.png").convert("RGB")
    s = min(WL / im.width, H / im.height)
    im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
    bg = Image.new("RGB", (WL, H), (255, 255, 255))
    bg.paste(im, ((WL - im.width) // 2, (H - im.height) // 2))
    p = WORK / f"left_{n}.png"
    bg.save(p)
    return p


def make_clip(n):
    anim = OUT / f"Page{n}_1440x1440_sfx.mp4"
    if not anim.exists():
        raise SystemExit(f"missing {anim}")
    clip = WORK / f"clip_{n}.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-i", anim, "-framerate", FPS, "-loop", "1",
         "-i", left_panel(n), "-filter_complex",
         "[0:v]scale=1440:1440,fps=30,format=yuv420p[a];[1:v]fps=30,format=yuv420p[l];"
         "[l][a]hstack=inputs=2:shortest=1[v]",
         "-map", "[v]", "-map", "0:a"] + ENC + ["-shortest", clip])
    return clip


def curl(A, B, p):
    h, w = A.shape[:2]
    r = w * 0.10
    c = (w + r) - p * (w + 2.4 * r)
    x = np.arange(w, dtype=np.float64)
    out = B.astype(np.float32)
    sh = (x > c + r) & (x < c + 2.2 * r)
    if sh.any():
        k = 1 - 0.35 * (1 - (x[sh] - (c + r)) / (1.2 * r))
        out[:, sh] *= k[None, :, None]
    flat = x < c
    out[:, flat] = A[:, flat]
    xs1 = 2 * c + np.pi * r - x
    m1 = (x < c) & (xs1 < w) & (xs1 >= 0)
    if m1.any():
        src = A[:, xs1[m1].astype(int)].astype(np.float32)
        out[:, m1] = (0.55 * src + 0.45 * 255) * 0.97
    t = np.clip((x - c) / r, 0, 1)
    xs2 = c + r * (np.pi - np.arcsin(t))
    m2 = (x >= c) & (x <= c + r) & (xs2 < w) & (xs2 >= 0)
    if m2.any():
        src = A[:, xs2[m2].astype(int)].astype(np.float32)
        shade = (1.0 - 0.45 * t[m2])[None, :, None]
        out[:, m2] = (0.55 * src + 0.45 * 255) * shade
    return np.clip(out, 0, 255).astype(np.uint8)


def swish(path, dur):
    sr = 44100
    n = int(sr * dur)
    rng = np.random.RandomState(1)
    nz = rng.normal(0, 1, n)
    band = nz - np.convolve(nz, np.ones(300) / 300, mode="same")
    band = np.convolve(band, np.ones(12) / 12, mode="same")
    t = np.linspace(0, 1, n)
    sig = band * (np.sin(np.pi * t) ** 2) * 0.55
    sig = sig / (np.abs(sig).max() + 1e-9) * 0.5
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes((sig * 32767).astype(np.int16).tobytes())


def frame_of(clip, path, last):
    if last:
        run(["ffmpeg", "-y", "-loglevel", "error", "-sseof", "-0.2", "-i", clip, "-update", "1", "-q:v", "1", path])
    else:
        run(["ffmpeg", "-y", "-loglevel", "error", "-i", clip, "-frames:v", "1", path])


def make_transition(n, m, clip_n, clip_m):
    la = np.array(Image.open(WORK / f"left_{n}.png").convert("RGB"))
    lb = np.array(Image.open(WORK / f"left_{m}.png").convert("RGB"))
    fa, fb = WORK / f"last_{n}.png", WORK / f"first_{m}.png"
    frame_of(clip_n, fa, True)
    frame_of(clip_m, fb, False)
    ra = np.array(Image.open(fa).convert("RGB"))[:, WL:WL + WR].astype(np.float32)
    rb = np.array(Image.open(fb).convert("RGB"))[:, WL:WL + WR].astype(np.float32)
    dur = TRANS
    wav = WORK / "swish.wav"
    swish(wav, dur)
    tr = WORK / f"trans_{n}_{m}.mp4"
    nfr = int(dur * FPS)
    pr = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                           "-s", f"{WL + WR}x{H}", "-r", str(FPS), "-i", "-", "-i", str(wav)] + ENC +
                          ["-shortest", str(tr)], stdin=subprocess.PIPE)
    for i in range(nfr):
        p = i / (nfr - 1)
        e = p * p * (3 - 2 * p)
        left = curl(la, lb, e)
        right = ((1 - e) * ra + e * rb).astype(np.uint8)
        pr.stdin.write(np.hstack([left, right]).tobytes())
    pr.stdin.close()
    if pr.wait() != 0:
        raise SystemExit("transition encode failed")
    return tr


def main():
    clips = {}
    for n in PAGES:
        print("clip", n, flush=True)
        clips[n] = make_clip(n)
    seq = []
    for i, n in enumerate(PAGES):
        seq.append(clips[n])
        if i + 1 < len(PAGES):
            m = PAGES[i + 1]
            print("transition", n, "->", m, flush=True)
            seq.append(make_transition(n, m, clips[n], clips[m]))
    lst = WORK / "concat.txt"
    lst.write_text("".join(f"file '{p}'\n" for p in seq))
    final = OUT / f"Pages{PAGES[0]}-{PAGES[-1]}_2560x1440_youtube.mp4"
    run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy",
         "-movflags", "+faststart", final])
    print("SAVED:", final)
    subprocess.run(["open", str(final)])


main()
