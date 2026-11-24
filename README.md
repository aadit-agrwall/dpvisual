# Deep learning page animations

Chapter 6 pairs Manim scenes with English and Hindi narration. From a checkout with the project Python dependencies and FFmpeg installed, render a page with:

```sh
.venv/bin/python tools/render_chapter_page.py 162
```

Each chapter page has three source files in `scenes/chapter06/`:

- `page_162.py` defines the 3D scene and records a cut between narration segments.
- `page_162.narration.txt` contains the English dialogue.
- `page_162.narration.hi.txt` contains the Hindi dialogue. Use one line per scene. For an exchange, write `TONY: question | TOM: answer`.

The exporter reads `ELEVEN_API_KEY` from the environment or the local `.env`. Each speaker uses the voice selected in `lib3d/characters.json`. Local voice selections in `media/voice/` override the defaults. Generated speech is cached between formats.

The Instagram render is `media/output/instagram_9x16/Page162.mp4` (1080 × 1920). The YouTube render is `media/output/youtube_16x9/Page162.mp4` (1920 × 1080). Both contain selectable Hindi and English audio, with Hindi selected by default. Use `--visuals-only` for silent preview files. The project keeps rendered media and the API key local.
