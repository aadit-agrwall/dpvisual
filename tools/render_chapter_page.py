"""Render a chapter 6 page in both social formats with English and Hindi audio.

Run .venv/bin/python tools/render_chapter_page.py 162 from any directory.
Use --visuals-only to produce silent previews independently of ElevenLabs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
from page_duo import ROOT, SR, parse, read, tts, write

FORMATS = [
    ("portrait", "instagram_9x16", 1920),
    ("landscape", "youtube_16x9", 1080),
]


def narration(script, language, starts, duration, destination):
    scenes = parse(script)
    if len(scenes) != len(starts):
        raise ValueError(f"{script} has {len(scenes)} scenes for {len(starts)} cues")
    track = np.zeros(round(duration * SR), dtype=np.float32)
    for index, (start, end, lines) in enumerate(zip(starts, starts[1:] + [duration], scenes), 1):
        clips = [read(tts(line, character, language)) for character, line in lines]
        cursor = round((start + 0.15) * SR)
        available = round((end - 0.1) * SR) - cursor
        gap = round(0.14 * SR)
        length = sum(map(len, clips))
        budget = available - gap * (len(clips) - 1)
        if budget <= 0 or length > 1.3 * budget:
            raise ValueError(f"Shorten {language} dialogue for scene {index}")
        for clip in clips:
            if length > budget:
                # atempo preserves the speaker's pitch.
                result = subprocess.run(
                    ["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1",
                     "-i", "pipe:0", "-af", f"atempo={length / budget}", "-f", "f32le", "pipe:1"],
                    input=clip.tobytes(), capture_output=True, check=True)
                clip = np.frombuffer(result.stdout, dtype="<f4")
            if cursor + len(clip) > round(end * SR):
                raise ValueError(f"{language} dialogue crossed cut {index}")
            track[cursor:cursor + len(clip)] = clip
            cursor += len(clip) + gap
    write(destination, track)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=int)
    parser.add_argument("--visuals-only", action="store_true")
    args = parser.parse_args()
    page = args.page
    scene = ROOT / f"scenes/chapter06/page_{page}.py"
    for mode, folder, height in FORMATS:
        media = ROOT / "media/chapter_formats" / mode
        subprocess.run([sys.executable, "-m", "manim", "render", "--progress_bar", "none",
                        "--media_dir", str(media), str(scene), f"Page{page}"],
                       env={**os.environ, f"P{page}_MODE": mode}, cwd=ROOT, check=True)
        video = media / f"videos/page_{page}/{height}p30/Page{page}.mp4"
        dest = ROOT / "media/output" / folder
        dest.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest) as work:
            tmp = Path(work)
            args_ff = ["ffmpeg", "-nostdin", "-v", "error", "-i", str(video)]
            if args.visuals_only:
                name = f"Page{page}_visuals.mp4"
                args_ff += ["-map", "0:v"]
            else:
                name = f"Page{page}.mp4"
                duration = float(subprocess.check_output([
                    "ffprobe", "-v", "error", "-show_entries", "format=duration",
                    "-of", "csv=p=0", str(video)]))
                cuts = json.loads((ROOT / f"media/sfx/page_{page}_cuts.json").read_text())
                starts = [0.0] + [cut["t"] for cut in cuts]
                for lang, suffix in [("en", ".txt"), ("hi", ".hi.txt")]:
                    audio = tmp / f"{lang}.wav"
                    narration(scene.with_suffix(".narration" + suffix), lang, starts, duration, audio)
                    args_ff += ["-i", str(audio)]
                args_ff += ["-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:a", "aac",
                            "-metadata:s:a:0", "language=eng", "-metadata:s:a:1", "language=hin",
                            "-disposition:a:0", "0", "-disposition:a:1", "default"]
            subprocess.run(args_ff + ["-c:v", "copy", "-movflags", "+faststart", str(tmp / name)], check=True)
            (tmp / name).replace(dest / name)


if __name__ == "__main__":
    main()
