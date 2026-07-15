<div align="center">

# 🧠 AI Pipeline Local

### Running open-source AI image & video generation on a consumer **8 GB GPU** under Windows native

> Flux · Qwen-Image · Z-Image · LTX-2B · Wan2GP · SadTalker
>
> **Others write hype. I write what happens when the 47th OOM crashes on an 8 GB card - and how it finally works.**

---

| Hardware | Environment | Framework |
|:---:|:---:|:---:|
| **RTX 5060 Laptop** · 8 GB VRAM · sm_120 | **Windows 11** native · **CUDA cu132** | **ComfyUI 0.25.0** |
| **PyTorch 2.13.0+cu132** · **Python 3.13.14** | No WSL · No Docker required | GGUF + FP8 quantized models |

---

### ⚡ Status Board

| Pipeline | Status | Best For | Release |
|:---|:---:|:---|:---:|
| Flux.1-dev Q4_K_S GGUF | ✅ Running | Photorealistic images with realism LoRA | [v1.0 / v2.0 ↗](https://github.com/Ultramanmao/ai-pipeline-local/releases/tag/v1.0) |
| Qwen-Image (full 17B) | ✅ Running | Best open-source Chinese text rendering (12 GB+ GPU) | [v1.0 ↗](https://github.com/Ultramanmao/ai-pipeline-local/releases/tag/v1.0) |
| Qwen-Image Q3_K_S GGUF | ✅ Running | Chinese text rendering on 8 GB (~5.4 GB VRAM) | [v1.0 ↗](https://github.com/Ultramanmao/ai-pipeline-local/releases/tag/v1.0) |
| Z-Image Turbo FP8 | ✅ Running | High-res, fast 4-step generation | — |
| LTX-2B I2V | ✅ Running | Text-to-video with keyframe locking | — |
| Flux.2 Klein 4B FP8 | ✅ Running | Newer architecture, 8-step | — |
| Wan2GP | 🟢 Installed | Video generation Web GUI (Sage2 RTX50xx) | — |
| SadTalker | ✅ Running | Talking-head / digital human (8 GB) | — |

</div>

---

## 📦 Release Archive

Each release is a fully reproducible technical article: complete ComfyUI node chains, error logs, parameter tables, and workflow JSON.

| Tag | Date | Title | Read |
|:---|:---:|:---|:---:|
| **v1.0** | 2026-07-14 | [Flux GGUF + Qwen-Image on 8GB RTX 5060: Real Results, Real Pitfalls](docs/releases/v1.0_Flux-GGUF-Qwen-Image-on-8GB-RTX-5060.md) | Full write-up |
| v2.0 | 2026-07-14 | [Flux.1-dev GGUF: The Realism LoRA Makes or Breaks It](docs/releases/v2.0_Flux-GGUF-Realism-LoRA-Comparison.md) | Full write-up |
| v3.0 | *coming soon* | Z-Image Turbo FP8 — 1312×1312 4-step generation + DiffSynth namespace pitfall | Planned |
| v4.0 | *coming soon* | LTX-2B I2V — 8 GB optimal config + keyframe locking + 3-segment relay | Planned |

---

## 🔧 Quick Start

Every pipeline runs under the same hardware stack. The one thing that matters:

```bash
# Always start ComfyUI with these flags on 8 GB GPUs:
python main.py --listen 0.0.0.0 --port 8188 --fp8_e4m3fn-unet --lowvram
```

Without `--lowvram`, T5XXL fp8 (5 GB) + a GGUF model (6.8 GB) + dequantization peak will OOM within seconds.

### Model Files

All model files live under `E:/WSL/ComfyUI/models/` (Windows native):

```
E:/WSL/ComfyUI/models/
├── unet/
│   ├── flux1-dev-Q4_K_S.gguf              # 6.8 GB
│   ├── Qwen_Image_Distill-Q3_K_S.gguf     # 8.95 GB
│   └── ...
├── text_encoders/
│   ├── Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf # 4.4 GB
│   ├── clip_l.safetensors
│   └── t5xxl_fp8_e4m3fn.safetensors
├── vae/
│   ├── ae.safetensors
│   └── qwen_image_vae.safetensors         # 243 MB
├── loras/
│   └── flux-realism-lora.safetensors
└── diffusers/
    └── Qwen-Image-2512/                    # 56 GB full model
```

---

## 📋 Key Lessons

| Lesson | Detail |
|:---|:---|
| **Always use `--lowvram`** | Mandatory on 8 GB with Flux / Qwen. No exceptions. |
| **Realism LoRA is the single most important Flux decision** | Without it, output looks like CGI. Bumping steps doesn't fix it. |
| **Qwen-Image needs Qwen2-VL, not Qwen3** | Hidden dimension mismatch: 2560 ≠ 3584. Causes silent failures. |
| **3-bit GGUF is blurry** | Q3_K_S works on 8 GB but quantization caps the detail. |
| **CFG=5 + normal scheduler** is Qwen-Image's sweet spot | Higher CFG or karras actually hurts clarity. |
| **`--output-directory` is ignored** | ComfyUI hardcodes output in `folder_paths.py` (line 63). |

---

## 🗺 Roadmap

| Stage | Target | What |
|:---|:---|:---|
| ~~v2.0 — Flux LoRA comparison~~ | ✅ Done |
| **This week** | v3.0 | Z-Image Turbo — 4-step vs 20-step + DiffSynth namespace pitfall |
| **Next** | v4.0 | LTX-2B I2V — 8GB config + keyframe locking + STG/StatNorm pitfall |
| **2 weeks** | Pages | `Ultramanmao.github.io` — aggregated blog homepage |

---

## 💬 Discuss

Have questions, hit a different error, or want to share your results? Open an [Issue ↗](https://github.com/Ultramanmao/ai-pipeline-local/issues).

## ⭐ Star

Star this repo to get notified when new pipelines are added and new releases go live.

---

## 📚 References

| Resource | Link |
|:---|:---|
| Flux.1-dev GGUF (city96) | https://github.com/city96/FLUX.1-dev-gguf |
| Qwen-Image (Alibaba) | https://github.com/QwenLM/Qwen-Image |
| Qwen-Image-2512 (HF) | https://huggingface.co/Qwen/Qwen-Image-2512 |
| ComfyUI-GGUF | https://github.com/comfyanonymous/ComfyUI-GGUF |
| DiffSynth-Studio | https://github.com/modelscope/DiffSynth-Studio |

---

## License

[MIT](LICENSE)
