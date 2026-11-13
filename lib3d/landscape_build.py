import subprocess, sys, wave, os
import numpy as np
from pathlib import Path
from PIL import Image

LW, H, RW, CURL = 1120, 1440, 1440, 1.8
FPS = int(os.environ.get("FPS", "30"))
PAGES = [int(a) for a in sys.argv[1:]]
L = Path("_land")
OUT = Path(f"assets/user-data/output/pages_{PAGES[0]}_{PAGES[-1]}_landscape_2k.mp4")
AF = "highpass=f=80,aecho=0.8:0.45:60|120:0.18|0.10,alimiter=limit=0.9,apad"
ENC = ["-c:v", "libx264", "-preset", "medium", "-crf", "16", "-pix_fmt", "yuv420p", "-r", str(FPS),
       "-video_track_timescale", "15360", "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "44100"]

def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print("FAILED:", " ".join(cmd[:8]), "\n", r.stderr[-1500:]); raise SystemExit(1)
    return r.stdout

def dur(p):
    return float(run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]).strip())

def size(p):
    return run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(p)]).strip()

def page_img(p):
    im = Image.open(f"assets/bookpage/page_{p}.png").convert("RGB"); w, h = im.size
    im = im.crop((6, 6, w - 6, h - 6)); w, h = im.size
    s = min((LW - 24) / w, (H - 24) / h)
    im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    cv = Image.new("RGB", (LW, H), (255, 255, 255))
    cv.paste(im, ((LW - im.width) // 2, (H - im.height) // 2))
    return cv

def curl_frame(A, B, t):
    _, W, _ = A.shape
    r = 78.0; s = t * t * (3 - 2 * t); f = W - (W + r) * s
    xs = np.arange(W, dtype=np.float32)
    Lc = max(W - f, 0.0); pr = np.pi * r
    fade = float(np.clip((f + r) / (2 * r), 0, 1))
    out = B.copy()
    flat = xs < f
    out[:, flat] = A[:, flat]
    e = f - max(Lc - pr, 0.0)
    sh = np.ones(W, np.float32)
    m = flat & (xs < e)
    sh[m] = 1 - 0.28 * np.exp(-(e - xs[m]) / (0.035 * W))
    out *= sh[None, :, None]
    if Lc > pr:
        mb = (xs >= e) & (xs <= f)
        g = 0.97 - 0.12 * (f - xs[mb]) / max(f - e, 1.0)
        out[:, mb] = (246 * g)[None, :, None]
    mc = (xs > f) & (xs < f + r)
    u = (xs[mc] - f) / r
    valid = r * np.arcsin(np.clip(u, 0, 1)) <= Lc
    idx = np.where(mc)[0][valid]
    uu = u[valid]
    if len(idx):
        out[:, idx] = (246 * (0.72 + 0.26 * (1 - uu ** 1.5)))[None, :, None]
    xr = f + r * np.sin(min(Lc / r, np.pi / 2))
    ms = xs >= xr
    k = fade * min(1.0, Lc / r)
    if k > 0:
        out[:, ms] *= (1 - 0.30 * k * np.exp(-(xs[ms] - xr) / (0.05 * W)))[None, :, None]
    return np.clip(out, 0, 255).astype(np.uint8)

def make_curl_sfx(path):
    SR = 44100; d = CURL + 0.3; t = np.arange(int(d * SR)) / SR
    u = np.clip(t / CURL, 0, 1); f = 450 + 950 * u ** 1.3
    sw = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.sin(np.pi * u) ** 0.7 * 0.30
    tb = np.clip(t - CURL * 0.85, 0, None)
    bell = np.sin(2 * np.pi * 1568 * tb) * np.exp(-tb * 6) * (t > CURL * 0.85) * 0.25
    x = np.stack([sw + bell, sw + bell], axis=1)
    w = wave.open(str(path), "wb"); w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
    w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()

make_curl_sfx("/tmp/curl_sfx.wav")
parts = []
for i, p in enumerate(PAGES):
    c = L / f"content_{p}.mp4"
    if not c.exists():
        print("missing", c); raise SystemExit(1)
    d = dur(c)
    print(f"page {p}: content {size(c)}, {d:.1f}s")
    run(["python3", "lib3d/landscape_audio.py", str(p)])
    page_img(p).save(L / f"left_{p}.png")
    seg = L / f"seg_{p}.mp4"
    run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-framerate", str(FPS), "-i", str(L / f"left_{p}.png"),
         "-i", str(c), "-i", f"/tmp/page_{p}_sfx_fun.wav",
         "-filter_complex", f"[1:v]scale={RW}:{H}:flags=lanczos,setsar=1[r];[0:v]setsar=1[l];[l][r]hstack=inputs=2,format=yuv420p[v]",
         "-map", "[v]", "-map", "2:a", "-af", AF, "-t", f"{d:.3f}", *ENC, str(seg)])
    parts.append(seg)
    if i < len(PAGES) - 1:
        A = np.asarray(page_img(p), dtype=np.float32)
        B = np.asarray(page_img(PAGES[i + 1]), dtype=np.float32)
        n = int(CURL * FPS)
        cv = L / f"curl_{p}.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{LW}x{H}", "-r", str(FPS), "-i", "-",
               "-f", "lavfi", "-i", f"color=c=white:s={RW}x{H}:r={FPS}", "-i", "/tmp/curl_sfx.wav",
               "-filter_complex", "[0:v][1:v]hstack=inputs=2:shortest=1,format=yuv420p[v]",
               "-map", "[v]", "-map", "2:a", "-t", f"{n / FPS:.4f}", *ENC, str(cv)]
        pr = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        for j in range(n):
            pr.stdin.write(curl_frame(A, B, j / (n - 1)).tobytes())
        pr.stdin.close(); err = pr.stderr.read().decode(); pr.wait()
        if pr.returncode:
            print("CURL FAILED", err[-1500:]); raise SystemExit(1)
        parts.append(cv)
        print(f"curl {p} -> {PAGES[i + 1]} done")

lst = L / "list.txt"
lst.write_text("".join(f"file '{x.resolve()}'\n" for x in parts))
run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst), "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(OUT)])
print(run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height", "-show_entries", "format=duration", "-of", "default=nw=1", str(OUT)]))
print("FINAL:", OUT)
