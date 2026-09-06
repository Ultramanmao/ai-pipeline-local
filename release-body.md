# v27.0 — H3 vs LTX-2.3: Video Pipeline Showdown on 8GB GPU

**TL;DR:** Ran MiniMax H3 and LTX-2.3 head-to-head on an 8GB RTX 5060 with 5 identical test cases. LTX is **8× faster** (3 min vs 7 min per 5-second clip) with **identical short-clip quality**, but H3 owns long video (Motion Context, 40-60s) and native audio generation.

---

## The Question

Two serious open-source video diffusion pipelines, one 8GB GPU. Which one?

| | H3 INT8 ConvRot | LTX-2.3 distilled |
|---|---|---|
| Model size | 20 GB | 14 GB (Q4_K_M) |
| Steps | 20 (native) | 8 (distilled) |
| Speed (5s clip) | **7 min** | **3 min** |
| Long video | ✅ Motion Context 40-60s | ❌ Single segment |
| Native audio | ✅ One pass | ⚠️ Needs AV pipeline |
| VRAM peak | ~7.5 GB | ~6 GB |

## Speed: Consistent 8× Gap

| Case | H3 | LTX | Ratio |
|------|-----|------|-------|
| Static scene | ~25 min | ~3 min | ~8× |
| Human motion | 6.8 min | ~3 min | ~8× |
| Fast motion | 7.0 min | ~3 min | ~8× |
| I2V fidelity | 6.8 min | ~3 min | ~8× |
| Audio test | 7.1 min | ~5 min | ~5× |

The 8× ratio holds across all non-audio cases. It's baked into step count (20 vs 8).

## Quality: No Gap at 5 Seconds

| Dimension | H3 | LTX |
|-----------|-----|------|
| Overall quality | 3.5/5 | 3.5/5 |
| Detail sharpness | 3.5 | 3.5 |
| Color rendition | **4.0** | 3.5 |
| Motion smoothness | 3.5 | 3.5 |
| Temporal consistency | 3.5 | 3.5 |

H3 has slightly better color saturation. Otherwise identical for short clips.

## The Biggest Pitfall: LTX Audio Pipeline

LTX generates silent video by default. The fix requires a specific node chain most tutorials skip:

```
LTXAVTextEncoderLoader (NOT DualCLIPLoaderGGUF)
→ LTXVEmptyLatentAudio
→ LTXVConcatAVLatent
→ Sampler
→ LTXVSeparateAVLatent
→ VAEDecode + LTXVAudioVAEDecode
```

`DualCLIPLoaderGGUF` loads the video text encoder only — no audio embeddings, no audio output, no error message.

## Decision Matrix

| Scenario | Pick | Why |
|----------|------|-----|
| Rapid prototyping | LTX | 3 min concept |
| Under 10 seconds | Either | No quality gap |
| Over 10 seconds | H3 | Motion Context |
| With audio | H3 | One-pass, zero config |
| Final production | H3 | 20 steps, better color |

## Lessons

1. **Speed gap is structural** — 20 steps vs 8 steps, no optimization will close it
2. **Quality gap only appears beyond 5 seconds** — long video chaining is H3's real advantage
3. **LTX audio works but the pipeline is obscure** — `LTXAVTextEncoderLoader` is the key
4. **H3 native audio is a workflow win** — one node, one pass, perfect sync
5. **VRAM headroom favors LTX** — 6 GB vs 7.5 GB, 1.5 GB matters on 8 GB cards

## Hardware Reference

| Spec | Value |
|------|-------|
| GPU | RTX 5060 Laptop 8 GB |
| RAM | 32 GB |
| CUDA | 12.8 |
| Framework | ComfyUI v0.34.1 |
| OS | Windows 11 native |
| H3 model | minimax_h3_fl2va_pruned_int8_convrot |
| LTX model | ltx-2.3-22b-distilled-1.1-Q4_K_M |
