# Flux.1-dev GGUF: The Realism LoRA Makes or Breaks It

> Date: 2026-07-14 · GPU: RTX 5060 Laptop 8GB · CUDA: cu132 · ComfyUI: 0.25.0
> A focused visual comparison of the single most important Flux decision.

## The One Question

I had Flux.1-dev Q4_K_S GGUF (6.8 GB) running on 8 GB VRAM. The model loaded, the nodes wired, ComfyUI started. But **should I bother with a LoRA, or just crank up the steps?**

Three iterations answered that in 3.5 hours.

---

## Iteration 1: 4 Steps, No LoRA

**Prompt:** "a cat sitting on a windowsill, sunlight, fluffy"
**Steps:** 4
**LoRA:** none
**CFG:** 1
**Sampler:** euler_ancestral

Result: cartoonish, soft, plastic-looking. Nothing like a photo. The model had barely started denoising. ❌

---

## Iteration 2: 28 Steps, No LoRA

**Prompt:** same "a cat..."
**Steps:** 28
**LoRA:** none
**CFG:** 1
**Sampler:** euler_ancestral

Result: slightly sharper than iteration 1, but still unmistakably CGI. The cat looked like a 3D render, not a real animal. The model was producing *something recognizable* but nothing photorealistic. ❌

I thought: "Maybe I just need more steps." 28 → 30 felt like a reasonable jump.

---

## Iteration 3: 30 Steps + flux-realism LoRA

**Prompt:**
```
photographic portrait of <subject>, shot on Sony A7R IV, 50mm f/1.4,
natural soft lighting, raw photo, depth of field, 85mm prime lens aesthetic
```
**Steps:** 30
**LoRA:** flux-realism-lora.safetensors (strength: 1.0)
**CFG:** 1
**Sampler:** euler_ancestral

Result: completely different image. Hair detail sharp. Lighting physically correct — warm falloff, natural specular on surfaces, soft depth of field. Catchlights in eyes that make it read as real. Comparable to Sony A7R IV output. ✅

**~100 seconds per 1024×1024 image.**

---

## The Verdict in One Sentence

> **The realism LoRA is the single most important Flux decision. Steps do not fix what the LoRA fixes.**

Bumping from 28 to 30 steps made almost no perceptible difference. Bumping from "no LoRA" to "realism LoRA" made everything.

The model doesn't know what a photograph is unless you tell it.

---

## The Complete SOP

```
DualCLIPLoader:
  clip_name1: clip_l.safetensors
  clip_name2: t5xxl_fp8_e4m3fn.safetensors
  type: flux

UnetLoaderGGUF:
  unet_name: flux1-dev-Q4_K_S.gguf

LoraLoaderModelOnly:
  lora_name: flux-realism-lora.safetensors
  strength_model: 1.0

ModelSamplingFlux:
  max_shift: 1.15, base_shift: 0.5, 1024×1024

FluxGuidance: guidance 3.5

KSampler:
  cfg: 1, denoise: 1, steps: 30
  sampler: euler_ancestral, scheduler: normal

EmptyLatentImage: 1024×1024

VAELoader: ae.safetensors
```

### The Prompt That Actually Works

```
photographic portrait of <subject>, shot on Sony A7R IV, 50mm f/1.4,
natural soft lighting, raw photo, depth of field, 85mm prime lens aesthetic
```

The prompt is a prompt. The LoRA is the decision. Get the LoRA right and you can experiment with prompts. Get the prompt right and forget the LoRA — the result will look like CGI.

---

## The Three Pitfalls That Cost Me Hours

### 1. `type=flux` on CLIPLoader — doesn't work in ComfyUI 0.25
Single CLIPLoader with `type=flux` silently produces wrong output. ComfyUI 0.25 changed how the type parameter is routed. **Fix:** use DualCLIPLoader with clip_l + t5xxl_fp8, type set on each node.

### 2. KSampler negative cannot be an empty array
`[]` crashes with a shape mismatch. **Fix:** connect a ConditioningZeroOut node instead of passing an empty list.

### 3. GGUF nodes not loading on some ComfyUI ports
One of my ComfyUI instances (889 nodes) had no UnetLoaderGGUF node registered. Another instance (1061 nodes) loaded them fine. **Fix:** restart ComfyUI with ComfyUI-GGUF extension present. Confirm by checking node count — 1000+ means GGUF is loaded.

---

## The Cost of "Just Try It"

- Iteration 1 (4 steps, no LoRA): ~15 seconds wasted
- Iteration 2 (28 steps, no LoRA): ~70 seconds wasted
- Iteration 3 (30 steps, LoRA): ~100 seconds producing the actual image
- Debugging CLIPLoader type + KSampler negative: ~90 minutes

The technical bugs are expensive. The LoRA question is cheap. Answer the LoRA question first.

---

## Reproducibility

| Parameter | Value |
|-----------|-------|
| Model | flux1-dev-Q4_K_S.gguf (6.8 GB, city96/FLUX.1-dev-gguf) |
| LoRA | flux-realism-lora.safetensors |
| LoRA strength | 1.0 (full) |
| Resolution | 1024×1024 |
| Steps | 30 |
| CFG | 1 |
| Sampler | euler_ancestral |
| Scheduler | normal |
| ComfyUI flags | `--fp8_e4m3fn-unet --lowvram` |
| Time per image | ~100s (8 GB, lowvram) |
| ComfyUI port | 8190 (1061 nodes) |

---

*Written from a Windows 11 / RTX 5060 8GB machine. The numbers in this article are from actual runs, not theory.*
