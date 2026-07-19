# Wan2.2 on 8 GB: Five Fixes to Unstick an sm_120 GPU

> Date: 2026-07-19 · GPU: RTX 5060 Laptop 8GB (sm_120) · CUDA: cu132 · Framework: ComfyUI-WanVideoWrapper · OS: Windows 11 native
> Release: https://github.com/Ultramanmao/ai-pipeline-local/releases/tag/v12.0

## TL;DR

Wan2.2-TI2V-5B is a 5B-parameter text-to-image-to-video model that produces high-quality short videos. On paper it should run on an 8 GB RTX 5060 — the model is 5B, the VRAM budget fits. In practice, a single sampling step takes ~200 seconds. Three steps = 10 minutes. Eighty-one frames = 30 minutes. Unusable.

The pipeline works. The model loads. T5 encoding finishes. Sampling starts. Then it crawls at one step per three minutes.

The problem is not the model. The problem is a kernel that hasn't been optimized for the card you're running.

This article documents five fixes, the single most important one, and why an RTX 5060 (sm_120, Blackwell) is the most expensive GPU to run video models on in 2026.

**The one number that matters:** switching the attention backend from Triton fallback to PyTorch SDP drops a single step from ~200s to ~10–15s. A 10–20× improvement. That is the only fix you need to start with.

---

## What Is Wan2.2-TI2V-5B

Wan2.2-TI2V-5B (by Wan-AI, Tsinghua) is a text-to-image-to-video pipeline. It takes a prompt and optionally an image, and produces a short video at up to 832×480 resolution. It is one of the highest-quality open-source text-to-video models available in mid-2026.

The key architectural choices:
- 5B parameters — roughly half the size of Wan2.1, designed for faster inference
- T5-XXL text encoder — native English prompt understanding
- Diffusion transformer with temporal attention regularization
- Native resolution: 832×480, 24 fps, 81 frames per video

On a 12 GB+ card (RTX 4090, A6000, A5000) it runs at 5–15 seconds per step. On an 8 GB RTX 5060, it runs at 200 seconds per step. The model is the same. The GPU is different.

---

## Why It Runs Slow on sm_120

The RTX 5060 is Blackwell (sm_120), a brand-new architecture. Triton, the high-performance attention kernel used by most open-source diffusion models, has not yet fully optimized for sm_120.

What happens:

| GPU | sm | Triton attention kernel | Per-step time |
|-----|----|------------------------|--------------|
| A6000/A5000 | sm_86/89 | Optimized | ~5–10s |
| RTX 4090 | sm_89 | Optimized | ~10–15s |
| RTX 5060 | sm_120 | **Fallback (unoptimized)** | **~200s** |

The fallback path is PyTorch's default eager-mode attention + a generic CUDA kernel. It is not broken. It is just slow — by 10–20×.

This is a known issue. The Triton team has added sm_120 support in Triton ≥ 3.1.0. FlashAttention 2.7+ supports Blackwell. But most of the open-source video models you install in 2026 ship with default configurations that pick Triton first — and get the fallback on sm_120.

**Rule: if you have an RTX 5060/5070/5080/5090, the attention backend is your bottleneck, not the model size.**

---

## Five Fixes (In Order of Impact)

### Fix 1: Switch attention_mode to sdp (Most important — 10–20×)

This is the single fix that changes everything.

WanVideoComfy and WanVideoWrapper expose an `attention_mode` parameter with four options: `triton`, `flash`, `xformers`, `sdp`.

**Use `sdp` (PyTorch SDP Attention).** On sm_120, PyTorch's `sdpa` (scaled dot-product attention) calls CUTLASS native kernels that are already optimized for Blackwell. It is faster than the Triton fallback by 10–20× on this card.

In ComfyUI, set `attention_mode="sdp"` on the model loading or sampling node. If the node doesn't expose the option, you may need to patch the WanVideoWrapper config directly.

If `sdp` is not available in your node, `xformers` (version 24.4+) is the second choice. It has also added sm_120 support.

| attention_mode | Per-step time on sm_120 | Notes |
|---------------|------------------------|-------|
| `triton` (default) | ~200s | Fallback kernel — avoid |
| `flash` | ~200s | FlashAttention fallback on sm_120 |
| `xformers` | ~10–20s | Works, but needs xformers ≥ 24.4 |
| **`sdp`** | **~10–15s** | **Recommended — CUTLASS native on sm_120** |

---

### Fix 2: Lower resolution + lower steps (Direct speed gain)

The default Wan2.2 resolution is 832×480 at 30–50 steps. Both are too high for a first pass on sm_120.

Recommended starting point: **480×360 at 10 steps**.

Why this works:
- The video model produces coherent motion at 480×360. Quality loss is visible but acceptable for a draft.
- Fewer pixels → faster attention computation → compounding speed gain.
- 10 steps is the minimum for a recognizable video. Going below 10 produces noisy output.

Pair with super-resolution upscaling as a post-processing step to recover quality.

---

### Fix 3: FP8 quantization (1.5–2× speed, half VRAM)

sm_120 has native FP8 Tensor Core support. WanVideoWrapper's ControlNet node exposes FP8 options (`fp8_e4m3fn`, `fp8_e4m3fn_fast`).

Applying FP8 to the UNet halves memory usage and gives a 1.5–2× speedup. The quality loss at 480×360 is minimal.

Stacking Fix 1 (sdp) + Fix 3 (FP8) is the recommended combination for stable 8 GB operation.

---

### Fix 4: Upgrade Triton / FlashAttention

Triton ≥ 3.1.0 has native sm_120 support. FlashAttention 2.7+ supports Blackwell.

This fix is the most permanent — once the upstream kernels are optimized, the fallback problem disappears entirely. But it requires a clean environment rebuild and is not the fastest path to a working pipeline.

If you already have `attention_mode="sdp"` working, skip this. The sdp path is already fast enough.

---

### Fix 5: torch.compile

```python
torch.set_float32_matmul_precision('high')
```

Applied to the UNet inference module. Works as a compound accelerator with sdp + xformers. The speedup is small on its own but adds up when stacked with Fix 1.

---

## The Working Configuration

After applying Fixes 1 + 3, a working Wan2.2-TI2V-5B run on an 8 GB RTX 5060 looks like this:

| Parameter | Value |
|-----------|-------|
| attention_mode | **sdp** |
| Resolution | **480×360** |
| Steps | 10 |
| Precision | FP8 (fp8_e4m3fn) |
| Model | Wan2.2-TI2V-5B |
| Text encoder | T5-XXL fp8 |
| VAE | Wan2.2 VAE |
| VRAM peak | ~7 GB (fits 8 GB) |
| Per-step time | **~10–15s** (vs ~200s default) |
| Total time (81 frames) | ~10–15 minutes (vs 30+ minutes default) |

The pipeline produces working output. The video is not broadcast-quality at 480×360 — but it is usable for drafts, short clips, and A/B testing prompts.

---

## The Three Lessons

| Lesson | Detail |
|--------|--------|
| **sm_120 is the new bottleneck** | Blackwell GPUs are newer than most open-source kernel stacks. Your card being "faster" doesn't mean models are faster on it. Always check the attention backend. |
| **sdp beats Triton on sm_120** | PyTorch's SDP path uses CUTLASS native kernels. Triton fallback does not. On sm_120, always set `attention_mode="sdp"`. |
| **5 fixes, but only 1–3 matter** | Fix 1 (sdp) is the unlock. Fixes 3 (FP8) and 2 (lower resolution) are compound gains. Fixes 4 (Triton upgrade) and 5 (torch.compile) are optional polish. |

---

## The Real Story

Wan2.2 is a good model. The VRAM fits. The architecture is sound. But open-source AI is a moving target: every new GPU architecture is a lagging variable in the kernel optimization pipeline.

On an RTX 4090 you wouldn't think twice about running Wan2.2. On an RTX 5060, you need to know about `attention_mode` before you start.

This is the pattern that will repeat with every new GPU architecture until the kernel stacks catch up. For sm_120 in 2026, the fix is already known. The cost of not knowing it is 200 seconds per step.

---

## Hardware Reference

| Spec | Value |
|------|-------|
| GPU | RTX 5060 Laptop, 8 GB, sm_120 |
| CUDA | cu132 (PyTorch 2.13.0+cu132) |
| ComfyUI | v0.24.0 |
| WanVideoWrapper | Latest (ComfyUI custom node) |
| Model | Wan2.2-TI2V-5B |
| Text encoder | T5-XXL fp8 |
| OS | Windows 11 native |
| Python | 3.13.14 |
| Status | ⚠️ Usable with sdp + FP8 + low resolution |

---

*Part of the [ai-pipeline-local](https://github.com/Ultramanmao/ai-pipeline-local) series — running open-source AI on a consumer 8 GB GPU.*
