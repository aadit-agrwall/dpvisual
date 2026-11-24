#!/usr/bin/env python3
"""Tom: voice-driven cartoon overlay with blinking eyes and lip-flap.

  python tools/tom.py rig                      # pose check image (closed / open / blink)
  python tools/tom.py demo                     # 6 s test with synthetic voice
  python tools/tom.py compose BASE.mp4 VOICE.wav [OUT.mp4]

Env: TOM_SVG  TOM_H=380  TOM_MARGIN=24  BASE_VOL=0.35
     TOM_EYES="15,16,17,18"  TOM_MOUTH="11,12"  TOM_BROW="14"   (shape numbers from tom_parts)
"""
import copy, json, math, os, pathlib, subprocess, sys, tempfile, wave
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]
SVG_DIR = ROOT / "assets" / "svg"
CACHE = ROOT / "media" / "tom"
OUT = ROOT / "assets" / "user-data" / "output"
SHAPES = {"path", "circle", "ellipse", "rect", "polygon", "polyline", "line"}
ET.register_namespace("", "http://www.w3.org/2000/svg")
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def find_svg():
    env = os.environ.get("TOM_SVG")
    if env:
        return pathlib.Path(env)
    svgs = sorted(p for p in SVG_DIR.iterdir() if p.suffix.lower() == ".svg")
    if not svgs:
        sys.exit(f"No .svg in {SVG_DIR}")
    for p in svgs:
        if "tom" in p.name.lower():
            return p
    return svgs[0]


def shapes(root):
    return [e for e in root.iter() if e.tag.split("}")[-1] in SHAPES]


def idx(name, default):
    return {int(x) for x in os.environ.get(name, default).replace(" ", "").split(",") if x}


def render_layer(tree, keep, name):
    t = copy.deepcopy(tree)
    for j, e in enumerate(shapes(t.getroot())):
        if j not in keep:
            e.set("style", (e.get("style") or "").rstrip(";") + ";display:none")
    CACHE.mkdir(parents=True, exist_ok=True)
    f, png = CACHE / f"layer_{name}.svg", CACHE / f"layer_{name}.png"
    t.write(f)
    try:
        subprocess.run(["rsvg-convert", "-h", "1000", str(f), "-o", str(png)], check=True)
    except FileNotFoundError:
        sys.exit("Need rsvg-convert (brew install librsvg)")
    return Image.open(png).convert("RGBA")


def load_rig(height):
    tree = ET.parse(find_svg())
    n = len(shapes(tree.getroot()))
    groups = {"eyes": idx("TOM_EYES", "15,16,17,18"),
              "mouth": idx("TOM_MOUTH", "11,12"),
              "brow": idx("TOM_BROW", "14")}
    moving = set().union(*groups.values())
    allset = set(range(n))
    full = render_layer(tree, allset, "full")
    bbox = full.getchannel("A").getbbox() or (0, 0, full.width, full.height)
    scale = height / (bbox[3] - bbox[1])
    size = (max(1, round((bbox[2] - bbox[0]) * scale)), height)

    def prep(img):
        return img.crop(bbox).resize(size, Image.LANCZOS)

    rig = {"base": prep(render_layer(tree, allset - moving, "base")), "size": size, "parts": {}}
    for name, keep in groups.items():
        im = prep(render_layer(tree, keep, name))
        bb = im.getchannel("A").getbbox()
        if bb:
            rig["parts"][name] = (im.crop(bb), bb[0], bb[1])
        else:
            print(f"WARN: layer {name} is empty, check shape numbers")
    return rig


def place(body, part, sx, sy, dx, dy, pivot):
    crop, x0, y0 = part
    px, py = pivot
    nw, nh = max(1, int(crop.width * sx)), max(1, int(crop.height * sy))
    nx, ny = px + (x0 - px) * sx + dx, py + (y0 - py) * sy + dy
    tmp = Image.new("RGBA", body.size, (0, 0, 0, 0))
    tmp.paste(crop.resize((nw, nh), Image.BICUBIC), (int(nx), int(ny)))
    return Image.alpha_composite(body, tmp)


def pose(rig, a, t, blink=None):
    """a = voice loudness 0..1, t = seconds. Returns RGBA image of Tom's head+body."""
    body = rig["base"].copy()
    H = rig["size"][1]
    P = rig["parts"]
    if "brow" in P:
        c, x0, y0 = P["brow"]
        body = place(body, P["brow"], 1, 1, 0, -0.02 * H * a, (x0 + c.width / 2, y0 + c.height / 2))
    if "mouth" in P:
        c, x0, y0 = P["mouth"]
        body = place(body, P["mouth"], 1 + 0.06 * a, 1 + 0.30 * a, 0, 0, (x0 + c.width / 2, y0))
    if "eyes" in P:
        c, x0, y0 = P["eyes"]
        if blink is None:
            ph = (t % 3.4) / 0.16
            blink = math.sin(math.pi * ph) if ph < 1 else 0.0
        body = place(body, P["eyes"], 1, 1 - 0.92 * blink, 0, 0, (x0 + c.width / 2, y0 + c.height / 2))
    if os.environ.get("TOM_FLIP", "1") == "1":
        body = body.transpose(Image.FLIP_LEFT_RIGHT)
    return body


def voice_envelope(voice, fps, n):
    with tempfile.TemporaryDirectory() as td:
        tmp = pathlib.Path(td) / "v.wav"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(voice), "-ac", "1",
                        "-ar", "16000", "-sample_fmt", "s16", str(tmp)], check=True)
        with wave.open(str(tmp)) as wf:
            x = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
    spf = 16000 / fps
    env = np.zeros(n, dtype=np.float32)
    for i in range(n):
        seg = x[int(i * spf):int((i + 1) * spf)]
        if len(seg):
            env[i] = math.sqrt(float(np.mean(seg * seg)))
    ref = np.percentile(env[env > 0], 95) if (env > 0).any() else 1.0
    env = np.clip(env / max(ref, 1e-6), 0, 1)
    sm = np.zeros_like(env)
    for i in range(n):
        prev = sm[i - 1] if i else 0.0
        sm[i] = env[i] if env[i] > prev else prev * 0.6 + env[i] * 0.4
    return sm


def probe(path):
    j = json.loads(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)]))
    s = j["streams"][0]
    a, b = s["r_frame_rate"].split("/")
    has_audio = bool(subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
         "stream=codec_type", "-of", "csv=p=0", str(path)]).strip())
    return int(s["width"]), int(s["height"]), float(a) / float(b), float(j["format"]["duration"]), has_audio


def compose(base, voice, out=None, height=None):
    height = height or int(os.environ.get("TOM_H", 380))
    margin = int(os.environ.get("TOM_MARGIN", 24))
    base_vol = float(os.environ.get("BASE_VOL", 0.10))
    voice_vol = float(os.environ.get("VOICE_VOL", 0.85))
    lp = int(os.environ.get("SFX_LP", 3500))
    duck = float(os.environ.get("SFX_DUCK", 6))
    base, out = pathlib.Path(base), pathlib.Path(out or OUT / (pathlib.Path(base).stem + "_tom.mp4"))
    W, H, fps, dur, has_a = probe(base)
    n = int(round(dur * fps))
    env = voice_envelope(voice, fps, n) if voice else np.zeros(n, dtype=np.float32)
    rig = load_rig(height)
    tw0, th0 = rig["size"]
    cw, ch = int(tw0 * 1.3), int(th0 * 1.2)

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(base),
           "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{cw}x{ch}", "-r", f"{fps}", "-i", "-"]
    if voice:
        cmd += ["-i", str(voice)]
    fc = f"[0:v][1:v]overlay=main_w-overlay_w-{margin}:main_h-overlay_h-{margin // 2}:shortest=1[v]"
    if voice and has_a:
        fc += (f";[0:a]highpass=f=150,lowpass=f={lp},volume={base_vol}[a0]"
               f";[2:a]apad,volume={voice_vol},asplit=2[v1][v2]"
               f";[a0][v1]sidechaincompress=threshold=0.03:ratio={max(duck, 1)}:attack=15:release=350[a0d]"
               f";[a0d][v2]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95[a]")
        amap = ["-map", "[a]"]
    elif voice:
        fc += ";[2:a]apad[a]"
        amap = ["-map", "[a]"]
    elif has_a:
        amap = ["-map", "0:a"]
    else:
        amap = []
    cmd += ["-filter_complex", fc, "-map", "[v]", *amap, "-t", f"{dur:.3f}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", str(out)]

    out.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for i in range(n):
            a, t = float(env[i]), i / fps
            tom = pose(rig, a, t)
            br = 0.012 * math.sin(2 * math.pi * 0.5 * t)
            sx, sy = 1 - 0.035 * a - br * 0.5, 1 + 0.07 * a + br
            hop = int(0.05 * th0 * a)
            tilt = 2.5 * math.sin(2 * math.pi * 0.7 * t) * (0.3 + a)
            tw, th = max(1, int(tw0 * sx)), max(1, int(th0 * sy))
            t_img = tom.resize((tw, th), Image.BICUBIC)
            canvas = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
            canvas.paste(t_img, ((cw - tw) // 2, ch - th - 6 - hop), t_img)
            canvas = canvas.rotate(tilt, resample=Image.BICUBIC, center=(cw / 2, ch - 6))
            p.stdin.write(canvas.tobytes())
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait() != 0:
        sys.exit("ffmpeg failed")
    print("WROTE", out)
    return out


def rig_check():
    rig = load_rig(int(os.environ.get("TOM_H", 380)))
    poses = [("closed", pose(rig, 0.0, 0, blink=0.0)),
             ("mouth open", pose(rig, 1.0, 0, blink=0.0)),
             ("blink", pose(rig, 0.0, 0, blink=1.0))]
    w, h = rig["size"]
    sheet = Image.new("RGBA", (w * 3 + 40, h + 20), "white")
    for i, (_, im) in enumerate(poses):
        sheet.alpha_composite(im, (10 + i * (w + 10), 10))
    f = CACHE / "rig_check.png"
    sheet.convert("RGB").save(f)
    print("WROTE", f, "parts:", list(rig["parts"]))
    return f


def demo():
    OUT.mkdir(parents=True, exist_ok=True)
    base, voice = CACHE / "demo_base.mp4", CACHE / "demo_voice.wav"
    CACHE.mkdir(parents=True, exist_ok=True)
    sr, secs = 16000, 6
    t = np.arange(sr * secs) / sr
    rng = np.random.default_rng(1)
    syll = np.clip(np.sin(2 * math.pi * 3.5 * t) + 0.3 * np.sin(2 * math.pi * 1.1 * t), 0, None)
    syll *= (np.sin(2 * math.pi * 0.4 * t) > -0.2)
    sig = (syll * (np.sin(2 * math.pi * 180 * t) + 0.3 * rng.standard_normal(len(t))) * 9000).astype(np.int16)
    with wave.open(str(voice), "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(sr); wf.writeframes(sig.tobytes())
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"color=white:s=2560x1440:r=30:d={secs}", "-pix_fmt", "yuv420p", str(base)], check=True)
    return compose(base, voice, OUT / "tom_demo.mp4")


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if c == "rig":
        rig_check()
    elif c == "demo":
        demo()
    elif c == "compose" and len(sys.argv) >= 4:
        compose(sys.argv[2], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    else:
        sys.exit(__doc__)
