# Deep learning page animations

Chapter 6 pairs Manim scenes with English and Hindi narration. From a checkout with the project Python dependencies, FFmpeg and rsvg-convert installed, render a page with:

```sh
.venv/bin/python tools/render_chapter_page.py 162
```

Each chapter page has three source files in `scenes/chapter06/`:

- `page_162.py` defines the 3D scene and records a cut between narration segments.
- `page_162.narration.txt` contains the English dialogue.
- `page_162.narration.hi.txt` contains the Hindi dialogue. Use one line per scene. For an exchange, write `TONY: question | TOM: answer`.

Place the project character artwork in `assets/svg/tom.svg` and `assets/svg/tony.svg`. The exporter animates Tony and Tom in the lower-right corner using each speaker’s own audio.

The exporter reads `ELEVEN_API_KEY` from the environment or the local `.env`. Each speaker uses the voice selected in `lib3d/characters.json`. Local voice selections in `media/voice/` override the defaults. Generated speech is cached between formats.

Instagram exports two single-audio editions, `Page162_hi.mp4` (Hindi/Hinglish) and `Page162_en.mp4` (English), in `media/output/instagram_9x16/` at 1080 × 1920. The original `Page162.mp4` path also plays Hindi. YouTube exports retain both selectable tracks at 1920 × 1080 in `media/output/youtube_16x9/`. Use `--visuals-only` for silent preview files. The project keeps rendered media and the API key local.

### Page 166 renderer

Page 166 uses ManimGX for the diagrams and Motion Canvas Core for the timing curves that drive title reveals and point transformations. Install its isolated runtimes once from the project root (Python 3.13+ and Node.js are required):

```sh
python3.14 -m venv media/render-tools/gx
media/render-tools/gx/bin/pip install manimgx==0.1.1
npm install --prefix media/render-tools/motion --ignore-scripts @motion-canvas/core@3.17.2 esbuild@0.28.2
```

Run `.venv/bin/python tools/render_chapter_page.py 166` with access to a GPU. The same dialogue-timed visual master feeds both language editions of each aspect ratio. Use `--visuals-only` to render without ElevenLabs.
