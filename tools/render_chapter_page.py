"""Render a chapter 6 page in both social formats with English and Hindi audio.

Run .venv/bin/python tools/render_chapter_page.py 162 from any directory.
Use --visuals-only to produce silent previews independently of ElevenLabs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile

import numpy as np
from page_duo import ROOT, SR, compose_duo, parse, read, tts, write

FORMATS = [
    ("portrait", "instagram_9x16", 1920),
    ("landscape", "youtube_16x9", 1080),
]


def spoken_clip(line, who, language):
    clip = read(tts(line, who, language, expressive=True))
    voiced = np.flatnonzero(np.abs(clip) > .008)
    if not len(voiced):
        raise ValueError(f"Empty narration from {who}")
    edge = round(.035 * SR)
    return clip[max(0, voiced[0]-edge):voiced[-1]+edge+1]


def beat_lengths(scene):
    lengths = []
    for lang, suffix in [("en", ".txt"), ("hi", ".hi.txt")]:
        beats = parse(scene.with_suffix(".narration" + suffix))
        lengths.append([sum(len(spoken_clip(line, who, lang)) / SR for who,line in beat)
                        for beat in beats])
    durations = []
    for english, hindi in zip(*lengths, strict=True):
        # Fit both editions to the same cut with a maximum 20% tempo increase.
        speech = max(max(english, hindi) / 1.20, min(english, hindi) / 1.04)
        durations.append(round(speech + .42, 2))
    return durations


def motion_track(events, duration, path):
    samples = np.zeros(round(duration * SR), dtype=np.float32)
    rng = np.random.default_rng(162)
    for when, kind in events:
        count = round(SR * (.22 if kind == "sweep" else .16))
        t = np.arange(count) / SR
        if kind == "sweep":
            sound = .032 * np.sin(2*np.pi*(380*t+1300*t*t)) * np.sin(np.pi*t/t[-1])**2
        else:
            sound = np.zeros(count)
            offsets = [0, .038, .08] if kind == "particles" else [0]
            for j, offset in enumerate(offsets):
                u = np.maximum(t-offset, 0)
                sound += .045*np.sin(2*np.pi*(1150+320*j)*u)*np.exp(-92*u)
            sound += .006*rng.normal(size=count)*np.exp(-65*t)
        start = round(when*SR)
        chunk = samples[start:start+count]
        chunk += sound[:len(chunk)]
    write(path, samples)


def timestamp(sample):
    millis = round(sample * 1000 / SR)
    seconds, ms = divmod(millis, 1000)
    minutes, sec = divmod(seconds, 60)
    hours, minute = divmod(minutes, 60)
    return f"{hours:02}:{minute:02}:{sec:02},{ms:03}"


def narration(script, language, starts, duration, destination):
    scenes = parse(script)
    if len(scenes) != len(starts):
        raise ValueError(f"{script} has {len(scenes)} scenes for {len(starts)} cues")
    subtitles = []
    tracks = {who: np.zeros(round(duration * SR), dtype=np.float32) for who in ("tom", "tony")}
    for index, (start, end, lines) in enumerate(zip(starts, starts[1:] + [duration], scenes), 1):
        clips = [spoken_clip(line, character, language) for character, line in lines]
        cursor = round((start + 0.07) * SR)
        available = round((end - 0.14) * SR) - cursor
        gap = round(0.11 * SR)
        length = sum(map(len, clips))
        budget = available - gap * (len(clips) - 1)
        if budget <= 0 or length > 1.3 * budget:
            raise ValueError(f"Shorten {language} dialogue for scene {index}")
        for (who, line), clip in zip(lines, clips):
            if length > budget:
                # atempo preserves the speaker's pitch.
                result = subprocess.run(
                    ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                     "-i", "pipe:0", "-af", f"atempo={length / budget}", "-f", "f32le", "pipe:1"],
                    input=clip.tobytes(), capture_output=True, check=True)
                clip = np.frombuffer(result.stdout, dtype="<f4")
            if cursor + len(clip) > round(end * SR):
                raise ValueError(f"{language} dialogue crossed cut {index}")
            subtitles.append(f"{len(subtitles)+1}\n{timestamp(cursor)} --> {timestamp(cursor+len(clip))}\n{who.title()}: {line}\n")
            tracks[who][cursor:cursor + len(clip)] = clip
            cursor += len(clip) + gap
    (destination / f"{language}.srt").write_text("\n".join(subtitles))
    files = []
    for who, track in tracks.items():
        path = destination / f"{language}-{who}.wav"
        write(path, track)
        files.append(path)
    return files


def publish(scene, video, dest, mode, tmp):
    duration = float(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "csv=p=0", str(video)]))
    cuts = json.loads((ROOT / f"media/sfx/{scene.stem}_cuts.json").read_text())
    starts = [0.0] + [cut["t"] for cut in cuts]
    events_path = ROOT / f"media/sfx/{scene.stem}_motions.json"
    effects = tmp / "motion.wav"
    motion_track(json.loads(events_path.read_text()) if events_path.exists() else [], duration, effects)
    outputs = {}
    for lang, suffix in [("en", ".txt"), ("hi", ".hi.txt")]:
        tracks = narration(scene.with_suffix(".narration" + suffix), lang, starts, duration, tmp)
        outputs[lang] = tmp / f"{lang}.mp4"
        compose_duo(video, *tracks, outputs[lang], 128 if mode == "portrait" else 170)
    # Each edition uses the matching character animation and primary narration.
    base = video.stem
    pan = "aformat=channel_layouts=stereo,pan=stereo|c0=0.225*c0+0.225*c1|c1=0.5*c0+0.5*c1"
    tags = {"en": "eng", "hi": "hin"}
    for lang, alternate, filename in [("hi", "en", base), ("en", "hi", base + "_en")]:
        languages = [lang] if mode == "portrait" else [lang, alternate]
        inputs, filters, maps = [], [], []
        for i, language in enumerate(languages):
            inputs.extend(["-i", str(outputs[language])])
            filters.append(f"[{i}:a]{pan}[v{i}];[v{i}][{len(languages)}:a]amix=inputs=2:duration=first:normalize=0[a{i}]")
            maps.extend(["-map", f"[a{i}]", f"-metadata:s:a:{i}", "language=" + tags[language],
                         f"-disposition:a:{i}", "default" if i == 0 else "0"])
        subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", *inputs, "-i", str(effects),
            "-filter_complex", ";".join(filters), "-map", "0:v", *maps,
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
            str(tmp / (filename + ".mp4"))
        ], check=True)
    for filename in [base, base + "_en"]:
        (tmp / (filename + ".mp4")).replace(dest / (filename + ".mp4"))
    # Refresh legacy Hindi filenames when a previous export used them.
    if mode == "portrait" or (dest / f"{base}_hi.mp4").exists():
        (tmp / "legacy_hi.mp4").hardlink_to(dest / f"{base}.mp4")
        (tmp / "legacy_hi.mp4").replace(dest / f"{base}_hi.mp4")
        shutil.copyfile(tmp / "hi.srt", dest / f"{base}_hi.srt")
    for language, filename in [("hi", base), ("en", base+"_en")]:
        (tmp / f"{language}.srt").replace(dest / f"{filename}.srt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=int)
    parser.add_argument("--visuals-only", action="store_true")
    args = parser.parse_args()
    page = args.page
    scene = ROOT / f"scenes/chapter06/page_{page}.py"
    timing = {} if args.visuals_only else {"DAEMON_BEATS": json.dumps(beat_lengths(scene))}
    for mode, folder, height in FORMATS:
        media = ROOT / "media/chapter_formats" / mode
        video = media / f"videos/page_{page}/{height}p30/Page{page}.mp4"
        env = {**os.environ, **timing, f"P{page}_MODE": mode}
        if page == 166:
            runtime = ROOT / "media/render-tools"
            motion = runtime / "motion"
            bundle = motion / "feature-map.cjs"
            curves = motion / "curves.json"
            subprocess.run([motion / "node_modules/.bin/esbuild", ROOT / "tools/motion/feature_map.mjs",
                            "--bundle", "--platform=node", f"--outfile={bundle}"],
                           env={**env, "NODE_PATH": str(motion / "node_modules")}, check=True)
            curves.write_bytes(subprocess.check_output(["node", bundle]))
            video.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([runtime / "gx/bin/python", scene, mode, video],
                           cwd=ROOT, env={**env, "MOTION_CURVES": str(curves)}, check=True)
        else:
            subprocess.run([sys.executable, "-m", "manim", "render", "--progress_bar", "none",
                            "--media_dir", str(media), str(scene), f"Page{page}"],
                           env=env, cwd=ROOT, check=True)
        dest = ROOT / "media/output" / folder
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest) as work:
            tmp = Path(work)
            if args.visuals_only:
                name = f"Page{page}_visuals.mp4"
                subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(video),
                                "-map", "0:v", "-c:v", "copy", "-movflags", "+faststart",
                                str(tmp / name)], check=True)
                (tmp / name).replace(dest / name)
            else:
                publish(scene, video, dest, mode, tmp)
                if mode == "landscape":
                    shutil.copyfile(scene.with_suffix(".narration.hi.txt"), dest / f"Page{page}_hi.transcript.txt")


if __name__ == "__main__":
    main()
