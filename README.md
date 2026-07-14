# AI Pipeline Local

Reproducible technical logs for running open-source AI image/video generation pipelines on a consumer 8 GB GPU (RTX 5060 Laptop) under Windows native.

> Others write hype. I write what happens when the 47th OOM crashes on an 8 GB card - and how it finally works.

## Pipelines Covered

| Pipeline | Status | Best For |
|----------|--------|----------|
| Flux.1-dev Q4_K_S GGUF | Runnable | Photorealistic images with LoRA |
| Qwen-Image (full 17B) | Runnable (12GB+ GPU) | Best open-source Chinese text rendering |
| Qwen-Image Q3_K_S GGUF | Runnable (8GB) | Budget alternative to full model |
| Z-Image Turbo FP8 | ComfyUI only | High-res, fast (4-step) |
| LTX-2B I2V | Runnable | Text-to-video, keyframe locking |
| Flux.2 Klein 4B FP8 | Runnable | Newer architecture, 8-step |

## Hardware

- GPU: NVIDIA RTX 5060 Laptop (8 GB VRAM, sm_120)
- OS: Windows 11 native (no WSL)
- CUDA: cu132 (RTX 50xx requires)
- Python: 3.13.14, PyTorch 2.13.0+cu132
- Framework: ComfyUI 0.25.0

## Releases

Each release is a reproducible technical article - full commands, errors, parameters, and workflow JSON.

- v1.0 - Flux GGUF + Qwen-Image on 8GB RTX 5060

## License

MIT
