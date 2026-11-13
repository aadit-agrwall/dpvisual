import sys, json, pathlib, subprocess, glob, os, importlib
sys.path.insert(0, str(pathlib.Path.cwd()))
n = sys.argv[1]
out = f"/tmp/page_{n}_sfx_fun.wav"
tl = pathlib.Path(f"media/sfx/Page{n}_timeline.json")
done = False
if tl.exists():
    try:
        m = importlib.import_module(f"scenes.chapter05.page_{n}")
        from scenes.chapter05.page_143 import write_wav
        data = json.load(open(tl))
        buf, slog = m.build_audio(data)
        write_wav(buf, pathlib.Path(out))
        print(f"audio page {n}: from timeline, {len(slog)} sounds, {data['total']:.1f}s")
        done = True
    except Exception as e:
        print(f"audio page {n}: timeline route failed ({e}), trying existing sfx video")
if not done:
    c = glob.glob(f"media/videos/**/Page{n}_1080x1080_sfx.mp4", recursive=True) + \
        glob.glob(f"assets/user-data/output/Page{n}_1080x1080_sfx.mp4")
    c = sorted(c, key=os.path.getmtime)
    if not c:
        print(f"FAIL: no timeline and no existing sfx video for page {n}"); sys.exit(1)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", c[-1], "-vn", "-ac", "2", "-ar", "44100", out], check=True)
    print(f"audio page {n}: extracted from {c[-1]}")
