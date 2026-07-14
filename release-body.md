# Running Flux GGUF + Qwen-Image on 8GB RTX 5060: Real Results, Real Pitfalls

> Date: 2026-07-14 | GPU: RTX 5060 Laptop 8GB | CUDA: cu132 | ComfyUI: 0.25.0 | OS: Windows 11 native

## TL;DR

Two pipelines that finally work on an 8 GB consumer GPU in 2026:

- **Flux.1-dev Q4_K_S GGUF** (6.8 GB) - photorealistic images, ~100s per 1024x1024, needs a realism LoRA or it looks like CGI
- **Qwen-Image** (full 17B = 56 GB on disk, or Q3_K_S GGUF = 8.95 GB) - strongest open-source Chinese text rendering, but the quantized GGUF version is noticeably blurry

Both are reproducible. Full node chains and workflow JSON below.

---

## Environment

| Item | Value |
|------|-------|
| GPU | RTX 5060 Laptop, 8 GB VRAM, sm_120 |
| CUDA | cu132 (required for RTX 50xx) |
| PyTorch | 2.13.0+cu132 |
| Python | 3.13.14 |
| ComfyUI | 0.25.0 |
| OS | Windows 11 native |
| Start flags | `--fp8_e4m3fn-unet --lowvram` (mandatory on 8 GB) |

Running ComfyUI without `--lowvram` on this hardware causes OOM within seconds: T5XXL fp8 (5 GB) + GGUF model (6.8 GB) + dequantization peak easily exceeds 8 GB. The `--lowvram` flag is not optional.

---

## Part 1: Flux.1-dev Q4_K_S GGUF

### The Problem

Flux.1-dev in full fp16 is ~12.8 GB - completely unrunnable on 8 GB VRAM. The GGUF quantized version (6.8 GB) theoretically fits. Does it produce usable images?

### The Answer

Yes - but **only with a realism LoRA**. Three iterations told the whole story:

| Version | Steps | LoRA | Style | Verdict |
|---------|-------|------|-------|---------|
| 1 | 4 | none | Cartoon/CGI | No |
| 2 | 28 | none | Still CGI | No |
| 3 | 30 | flux-realism | Photorealistic | Yes |

**The realism LoRA is the single decisive factor.** Bumping steps from 28 to 30 made almost no difference.

### The SOP

```
DualCLIPLoader: clip_l.safetensors + t5xxl_fp8_e4m3fn.safetensors (type: flux)
UnetLoaderGGUF: flux1-dev-Q4_K_S.gguf
LoraLoaderModelOnly: flux-realism-lora.safetensors, strength_model: 1.0
ModelSamplingFlux: max_shift 1.15, base_shift 0.5, 1024x1024
FluxGuidance: 3.5
KSampler: cfg 1, denoise 1, steps 30, euler_ancestral, normal
EmptyLatentImage: 1024x1024
VAELoader: ae.safetensors
```

Photographic prompt template:

```
photographic portrait of <subject>, shot on Sony A7R IV, 50mm f/1.4,
natural soft lighting, raw photo, depth of field, 85mm prime lens aesthetic
```

Quality: hair detail sharp, leather highlights physically correct, warm lighting, depth of field, catchlights in eyes. Evaluated as comparable to Sony A7R IV output. ~100s per image on 8 GB with lowvram.

### Pitfalls (Flux)

1. **CLIPLoader type parameter** - ComfyUI 0.25 no longer accepts `type=flux` on a single CLIPLoader. Must use DualCLIPLoader (clip_l + t5xxl_fp8).
2. **SaveImage path hardcoded** - ComfyUI writes to its built-in output_dir, ignoring the `--output-directory` CLI flag (hardcoded in `folder_paths.py` line 63). Must manually move files or edit that line.
3. **OOM without `--lowvram`** - crashed at step 8 with 502 error. `--lowvram --fp8_e4m3fn-unet` stabilized it.
4. **GGUF nodes not loaded** - one ComfyUI port had no GGUF nodes registered. Restart with ComfyUI-GGUF extension loaded (1061+ nodes).
5. **KSampler negative cannot be empty** - `[]` crashes. Use ConditioningZeroOut node instead.

---

## Part 2: Qwen-Image

### The Model

Qwen-Image-2512 by Alibaba. 20B MMDiT with Qwen 2.5-VL 7B as its native text encoder. Apache 2.0. Standout feature: **open-source Chinese text rendering that actually works** - posters, comics, logos with correct Chinese characters.

Full model: ~56 GB on disk. Impossible on 8 GB without heavy optimization.

### Two Routes Tested

| | Full 17B | Q3_K_S GGUF |
|---|---|---|
| Model loader | DiffusersLoader | UnetLoaderGGUF |
| Text encoder | Full (built-in) | Qwen2.5-VL-7B Q4_K_M GGUF |
| VAE | Full (built-in) | qwen_image_vae.safetensors |
| Steps | 20 | 30 |
| CFG | 5.0 | 7.0 |
| Scheduler | normal | karras |
| Quality | **9/10** | 6-7/10 (blurry) |
| GPU | 12 GB+ | **8 GB works** |
| VRAM (GGUF) | - | ~5.4 GB |

### Full Model: Best Quality

**CFG=5 + normal scheduler + 20 steps.** Bumping CFG to 7.0 or switching to karras actually hurts clarity. 20 steps is enough - more steps give no visible improvement.

### GGUF Route: 8GB Friendly But Blurry

Q3_K_S (8.95 GB, ~5.4 GB VRAM) runs on 8 GB and produces recognizable images, but 3-bit quantization caps the detail. 50 steps does not help - that is the quantization ceiling.

```
UnetLoaderGGUF -> Qwen_Image_Distill-Q3_K_S.gguf
CLIPLoaderGGUF -> Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf (type=qwen_image)
VAELoader -> qwen_image_vae.safetensors
CLIPTextEncode -> Chinese prompt (positive)
CLIPTextEncode -> "ugly, blurry, text" (negative)
EmptyLatentImage -> 1024x1024
KSampler -> steps 20-30, cfg 4.0, sgm_uniform
VAEDecode -> SaveImage
```

Resolution support (GGUF): 768x768 (~10s/step), 1024x1024 (~12s/step), 1280x1280 (~14s/step).

### The Dimension Trap

When wiring up the GGUF route, I paired the model with Qwen3-4B Q4_K_M text encoder GGUF and got:

```
normalized_shape=[3584], expected input with shape [*3584],
but got input of size[1, 31, 2560]
```

**Root cause:** Qwen3-4B has hidden dimension **2560**. Qwen-Image expects **3584** (Qwen2-VL family). Different architectures, both called "Qwen".

**Fix:** Use `Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf` (type=qwen_image) or the native QwenLoader node.

### Pitfalls (Qwen-Image)

1. **Text encoder dimension mismatch** - Qwen3 (2560) != Qwen-Image (3584). Use Qwen2.5-VL, not Qwen3.
2. **Q4_K_M GGUF (13 GB) is too big for 8 GB** - estimated 7-8 GB VRAM, guaranteed OOM. Stick with Q3_K_S on 8 GB cards.
3. **Xet CDN drops files >9 GB** - the 13 GB Q4_K_M file would not fully download. HuggingFace mirror with `-4` (force IPv4) + HTTP proxy (not SOCKS5) required on this network.
4. **VAE misidentified as WanVAE** - ComfyUI matches the Qwen VAE by key name to WanVAE; compatible but not optimal.

---

## Side-by-Side Comparison

| Pipeline | VRAM | Chinese Text | Speed | Verdict |
|----------|------|-------------|-------|---------|
| SDXL | ~4-5 GB (stable) | English only | ~5-10s | Fastest |
| Z-Image Turbo FP8 | stable | mediocre | ~40s | High-res |
| Flux.2 Klein FP8 | stable | mediocre | ~40s | New architecture |
| Flux.1-dev GGUF | ~6 GB | mediocre | ~100s | Best photorealistic |
| Qwen-Image full | 12 GB+ | Best in class | slow | Best quality |
| Qwen-Image GGUF | ~5.4 GB | Strong | ~240s | 8GB only option |

---

## Verdict

- **Best photo pipeline on 8 GB:** Flux.1-dev Q4_K_S GGUF with flux-realism LoRA. No contest.
- **Best Chinese text on 8 GB:** Qwen-Image Q3_K_S GGUF - blurry but the only open-source model that renders Chinese text correctly.
- **Best overall quality:** Qwen-Image full 17B, but needs 12+ GB VRAM.
- **The realism LoRA is the single most important Flux decision** - without it, the output is unrecognizable.
- **Always use `--lowvram` on 8 GB cards with Flux/Qwen.** No exceptions.

---

## Model Files

| File | Size | Location |
|------|------|----------|
| flux1-dev-Q4_K_S.gguf | 6.8 GB | models/unet/ |
| Qwen_Image_Distill-Q3_K_S.gguf | 8.95 GB | models/unet/ |
| Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf | 4.4 GB | models/text_encoders/ |
| qwen_image_vae.safetensors | 243 MB | models/vae/ |
| flux-realism-lora.safetensors | - | models/loras/ |

---

## Reference Links

- Flux.1-dev GGUF: https://github.com/city96/FLUX.1-dev-gguf
- Qwen-Image: https://github.com/QwenLM/Qwen-Image
- Qwen-Image-2512: https://huggingface.co/Qwen/Qwen-Image-2512
- ComfyUI-GGUF: https://github.com/comfyanonymous/ComfyUI-GGUF
- DiffSynth-Studio: https://github.com/modelscope/DiffSynth-Studio

---

*Written from a Windows 11 / RTX 5060 8GB machine. All parameters verified by actual runs, not theory.*
