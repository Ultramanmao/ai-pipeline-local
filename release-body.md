# One Command, Text-to-Vertical-Video With Voiceover and Subtitles

> Date: 2026-07-24 · GPU: RTX 5060 Laptop 8GB (sm_120, Blackwell) · CUDA: cu132 · Framework: IndexTTS2 + FunASR + FFmpeg · OS: Windows 11 native
> Release: https://github.com/Ultramanmao/ai-pipeline-local/releases/tag/v16.0

## TL;DR

A text-to-video pipeline that takes a paragraph of Chinese text and outputs a 1080×1920 vertical MP4 with voiceover and burned-in subtitles — in one command, in about 30 seconds, on an 8 GB laptop GPU, with zero cloud services.

The pipeline: **Text → IndexTTS2 (RTF 0.70) → FunASR (RTF 0.08) → FFmpeg composition**.

The code: 172 lines, `full_chain.py`, copy and run.

## The Pipeline

| Stage | Tool | Time | RTF |
|-------|------|------|-----|
| IndexTTS2 voiceover | paraformer-zh reference audio | 15–16s | 0.70 |
| FunASR transcription | paraformer-zh, SRT output | 4s | 0.08 |
| FFmpeg composition | scale+pad + drawtext + audio merge | ~5s | — |
| **Total** | | **~30s** | — |

Output: `final.mp4`, 1.2 MB, 6s, 1080×1920 vertical. Subtitles, voiceover, video — all in one shot.

## The One-Command Interface

```bash
python full_chain.py "8GB graphics card, from text to full video in one shot" \
  --video your_video.mp4 \
  --out final.mp4
```

Three functions inside:

```python
def main():
    step1_tts(args.text, args.ref, audio)         # IndexTTS2 voiceover
    step2_stt(audio, srt)                         # FunASR transcription
    step3_compose(args.video, audio, srt_text, args.out)  # FFmpeg composition
```

## The Biggest Bug: FFmpeg's `:` vs. Windows' `E:/`

In FFmpeg's `filter_complex` syntax, the colon `:` is a parameter delimiter.

In a Windows path `E:/workout/pipeline/video.mp4`, the second character is a colon.

FFmpeg reads `E:` and treats it as a filter parameter. The entire filter chain blows up.

Five approaches tried:

| Approach | Result |
|----------|--------|
| `file:///` URI prefix | Not recognized by FFmpeg |
| ASS subtitle file | Colon still parsed as parameter |
| `drawtext` inline `text=` | Chinese characters mangled |
| `drawtext textfile=` | Path colon still explodes |
| **✅ All paths converted to relative** | Works |

The fix: every path argument to FFmpeg (input video, audio, output file) goes through a `_rp()` helper that converts absolute Windows paths to relative paths, with working directory pinned to the pipeline directory. No colons in paths, filter chain runs clean.

Secondary bug fixed: `step1_tts` originally called `os.chdir()` to switch directories for IndexTTS, which polluted FFmpeg's `cwd`. Switched to passing absolute paths directly to IndexTTS — `cwd` stays clean.

## Why This Matters

The individual tools — IndexTTS2, FunASR, FFmpeg — are off-the-shelf. The value is that **they are stitched into one line**, eliminating all the manual glue work between writing a script and getting a finished vertical video.

Full code: `E:/workout/pipeline/full_chain.py`

## Next

1. Feed LTX-2B I2V-generated clips into the chain for motion in the video layer
2. Long-text segmentation: auto-split, segment TTS, merge audio, transcribe, compose
3. OpenMontage packaging layer: cover cards, chapter dividers, end cards → RedNote-ready output

---

*Part of the [ai-pipeline-local](https://github.com/Ultramanmao/ai-pipeline-local) series — running open-source AI on a consumer 8 GB GPU.*
