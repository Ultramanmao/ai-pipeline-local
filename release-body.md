# Bonsai-27B on 8 GB: How I Fit a 27B LLM on an RTX 5060 Laptop

> Date: 2026-07-24 · GPU: RTX 5060 Laptop 8GB (sm_120, Blackwell) · CUDA: 13.3.73 · Framework: prismml-llama-cpp · OS: Windows 11 native
> Release: https://github.com/Ultramanmao/ai-pipeline-local/releases/tag/v15.0

## TL;DR

A 27B-parameter language model on an 8 GB laptop GPU. On paper this should not exist — the unquantized model weighs ~50 GB. The answer is extreme quantization: squeeze 27B down to 3.6 GB (Q1_0, 1-bit ternary quantization) and compile llama.cpp with custom Blackwell kernels.

| Mode | Prompt | Generation |
|------|--------|-----------|
| CPU-only | 8.2 t/s | 7.6 t/s |
| GPU (Blackwell) | **108.3 t/s** | **34.5 t/s** |
| GPU speedup | 13.2× | 4.5× |

GPU peak VRAM: ~5 GB. The model fits. The question is whether 1-bit quantization destroys the model — and whether compiling for a GPU that mainstream toolchains don't yet recognize is worth it.

---

## What Is Bonsai-27B

Bonsai-27B is a 27B-parameter dense language model. It's not a MoE. It's not flash-attention optimized for consumer cards. Out of the box, it needs ~50 GB of VRAM. On a 27B model, you're competing with every consumer GPU release since the 3080.

The quantization route that makes this work is PrismML's ternary quantization scheme: Q1_0_g128 (1.33 bits per weight) collapses the model to **3.53 GB**. A Q2_0 variant exists at ~6.7 GB with measurably better MMLU (+4.6 percentage points), but fits in 8 GB only at ~7 GB peak — too tight for comfortable use.

This means the model has to be loaded into an inference engine that understands ternary quantization. The mainstream llama.cpp fork does not. You need PrismML's fork, compiled with CUDA support for the GPU architecture your card uses.

---

## The GPU Architecture Problem

The RTX 5060 Laptop GPU uses the **sm_120** compute architecture (Blackwell). This is a first-generation Blackwell consumer card. It is not yet supported by the mainstream CUDA toolkit.

| Tool | Version | sm_120 Support |
|------|---------|----------------|
| CUDA 12.4 | NVCC 12.4 | ❌ `"Unsupported gpu architecture 'compute_120'"` |
| CUDA 12.8 | NVCC 12.8 | ⚠️ experimental, limited |
| CUDA 13.3 | NVCC 13.3.73 | ✅ native support |

CUDA 13.3 is the minimum viable toolkit. Without it, the compilation fails before it begins.

---

## Pitfall 1: CUDA 13.3 Requires Administrator Install

The CUDA 13.3 installer (2.53 GB) is a Windows executable that must write to `C:\Program Files\NVIDIA GPU Computing Toolkit\`. Running it headless via PowerShell `-Verb RunAs` in a background process fails silently — exit code 8 — because UAC (`EnableLUA=1`) silently rejects the elevated request from a background process.

The 8 GB CUDA download over a Chinese proxy runs at ~128 KB/s with frequent `schannel: server closed abruptly` disconnections. Single-threaded `curl -C -` (resume) is the only reliable approach. Multi-threaded concurrent downloads are fully rejected by most residential proxies (0 bytes returned).

**Fix:** Either double-click the installer and select Custom → Compiler only (~5 minutes), or use PowerShell with interactive RunAs. There is no clean background install path on UAC-enabled Windows.

After install, the CUDA 12.4 toolkit that shipped with the driver remains intact alongside 13.3 — both coexist without conflict.

---

## Pitfall 2: MSYS2 Bash Cannot Find the MSVC Compiler

PrismML's llama.cpp fork requires MSVC for CUDA compilation on Windows. MSYS2's `bash` environment does not inherit MSVC's `INCLUDE`, `LIB`, or `PATH`. CMake silently fails because it cannot find `cl.exe`.

**Fix:** Explicitly set `INCLUDE` and `LIB` to point to the Windows SDK and MSVC directories, and add the MSVC `bin` directory to `PATH`. Also set `CUDAHOSTCXX` explicitly — otherwise `nvcc` cannot find the host compiler and fails with `cl.exe not found`.

---

## Pitfall 3: The build.ninja Linux Flag Crashes the Linker

CMake reports success, but the `build.ninja` file contains:

```
LINK_FLAGS = -shared /machine:x64
```

The `-shared` flag is a GCC/LLVM linker flag. It does not exist in MSVC's `link.exe`. When Ninja invokes the link step, `link.exe` encounters an unknown flag and crashes with `ACCESS_VIOLATION` (exit code 3221225794).

**Fix:** Replace `-shared` with `/DLL`:

```
LINK_FLAGS = /DLL /machine:x64
```

One line. This is a CMake generator-platform mismatch — Ninja assumes Linux conventions on a Windows cross-build that targets MSVC. It is not documented anywhere upstream.

---

## Pitfall 4: Ternary Quantization Is Not Mainstream

The Bonsai-27B GGUF uses ternary quantization (Q1_0_g128 / Q2_0_g128). The mainstream llama.cpp and all its consumer forks (llama.cpp, qwen, mlx) do not understand ternary weights. Loading the model produces garbage or crashes.

The PrismML fork that supports ternary quantization is a separate repository (`aionoid/prismml-llama-cpp`) with a specific branch and commit. This is not a drop-in replacement — it's a different codebase that tracks the same upstream and adds ternary support.

---

## The Working Configuration

```bash
cmake -G Ninja ^
  -B build.gpu ^
  -S . ^
  -DLLAMA_CUDA=ON ^
  -DCMAKE_CUDA_ARCHITECTURES=120
```

Run in the PrismML fork at the correct commit. Compilation produces `llama-cli.exe` (62 MB with CUDA support, 7 MB CPU-only).

Runtime:

```bash
llama-cli.exe ^
  -m Bonsai-27B-Q1_0.gguf ^
  -p "Your prompt here" ^
  -n 512 ^
  -c 2048 ^
  --temp 0.7 ^
  --n-gpu-layers 64
```

All 64 layers load to CUDA0. Peak VRAM consumption: ~5 GB. Remaining VRAM (~3 GB): available for other models or system overhead.

---

## Performance

| Metric | CPU-only | GPU (Blackwell) | Speedup |
|--------|----------|-----------------|---------|
| Prompt processing | 8.2 t/s | **108.3 t/s** | 13.2× |
| Token generation | 7.6 t/s | **34.5 t/s** | 4.5× |
| Peak memory | ~4 GB RAM | ~5 GB VRAM | — |

Test conditions: `-n 64` output tokens, `-c 2048` context, temperature 0.7.

The prompt speedup is the more dramatic number — 13× over CPU. This is because prompt processing is memory-bandwidth bounded and the GPU's HBM-equivalent memory subsystem (GDDR7 on Blackwell) is massively faster than system RAM. Token generation is compute-bound and the speedup is smaller, but 34.5 t/s is fast enough for real-time interaction on a 27B model.

---

## Q1_0 vs Q2_0: The Trade-Off

| | Q1_0 (1.33 bits) | Q2_0 (1.73 bits) |
|---|---|---|
| File size | 3.53 GB | ~6.67 GB |
| Peak VRAM | ~5 GB (comfortable) | ~7 GB (tight) |
| MMLU (theoretical) | ~61.2 | ~65.8 (+4.6 pp) |
| HellaSwag | ~79.4 | ~81.1 (+1.7 pp) |
| 8 GB fits? | ✅ | ⚠️ (barely) |

The Q2_0 file is 6.67 GB. At download speeds of ~225 KB/s through residential proxies, the file takes 10+ hours with frequent disconnections. In practice, the Q2_0 GGUF I downloaded corrupted at the file level — an offset mismatch in the weight tensor (19.8 MB misalignment), not a build error. The file itself was damaged, not the binary.

For most use cases, Q1_0 at 3.6 GB is the practical choice. The model is coherent, chain-of-thought reasoning works, and 34.5 t/s is more than enough for an offline private model.

---

## Lessons

| Lesson | Detail |
|--------|--------|
| **Blackwell is a first-class citizen in CUDA 13.3 only** | NVCC 12.4 cannot see sm_120 at all. No workaround. |
| **Ternary quantization is the only path** | Mainstream GGUF does not support it. You need a specific fork. |
| **1-bit is viable for 27B** | At 3.6 GB, a 27B model runs on the cheapest 8 GB card. The quality loss is real but acceptable for offline use. |
| **MSVC + CMake + Ninja on Windows is brittle** | Three silent failure modes (MSVC not found, Linux linker flags, CUDAHOSTCXX missing) require manual patching that upstream has not fixed. |
| **Q2_0 is too tight for 8 GB** | 7 GB peak VRAM leaves no headroom. Q1_0 is the only comfortable choice. |
| **This is not an Agent mainline** | 34.5 t/s is fast enough for chat, not fast enough for sub-second agent loops. Position this as a privacy/offline fallback, not a primary inference engine. |

---

## The Real Story

A 27B language model on 8 GB should not work. It does, but only by collapsing the model to 1.33 bits per weight, using a CUDA version released less than a year ago, and compiling against a custom fork of llama.cpp that adds support for a quantization scheme the mainstream ignores.

The pipeline works. The 4.5× GPU speedup makes it usable. But the number of manual interventions required — administrator-privileged CUDA installation, MSVC path plumbing, linker flag patching, ternary quantization fork compilation — makes this a one-off engineering exercise, not a drop-in solution. If you want a private 27B on an 8 GB card, it's doable. If you want it to just work, pick a smaller model.

---

## Hardware Reference

| Spec | Value |
|------|-------|
| GPU | RTX 5060 Laptop, 8 GB, sm_120 (Blackwell) |
| Driver | NVIDIA-SMI 596.36 |
| CUDA | 13.3.73 |
| MSVC | 19.44.35207 |
| CMake | 4.4 |
| Python | 3.13.14 |
| OS | Windows 11 native |
| Model | Bonsai-27B-Q1_0.gguf (ternary, 3.53 GB) |
| Engine | prismml-llama-cpp (aionoid fork), commit 3aa406f |
| VRAM peak | ~5 GB |
| Prompt speed | 108.3 t/s |
| Generation speed | 34.5 t/s |

---

*Part of the [ai-pipeline-local](https://github.com/Ultramanmao/ai-pipeline-local) series — running open-source AI on a consumer 8 GB GPU.*
