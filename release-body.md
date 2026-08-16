# Gemma-4-12B-Instruct Q4_K_S on 8 GB: The Partial-Offload Gamble on a Laptop GPU

> Repo: [Ultramanmao/ai-pipeline-local](https://github.com/Ultramanmao/ai-pipeline-local)
> Benchmark hardware: AMD Ryzen 7 H 260 (16T) / RTX 5060 Laptop 8 GB / 32 GB RAM / sm_120 (Blackwell)
> Date: 2026-08-16

---

## TL;DR

A 6.76 GB Q4_K_S GGUF of **Gemma-4-12B-Instruct** (official Google instruct fine-tune)
on an RTX 5060 Laptop 8 GB. It runs — barely. Full offload (`-ngl 99`) OOMs.
Partial offload (`-ngl 28`) holds it at a steady **10 t/s generation speed**,
roughly one-third of the 35B MoE baseline that lives fully on GPU.

For a 12B model with an instruct fine-tune, the results are a mixed bag:
translation, classical Chinese, and math are clean. Code and logic get
truncated by the chain-of-thought block. Cultural knowledge has a notable
hallucination. Total score **34/60**.

This is not a主力 model. But for tasks where a 7B is not enough
and a 35B is too slow, it fills a real gap.

---

## Why this model

**Model**: `google/gemma-4-12b-it` (Q4_K_S, 6.76 GB, standard GGUF format)
- Official Google instruct fine-tune (not a community quantization)
- Standard GGUF — loads with the stock ggml-org llama.cpp build,
  no prismml fork, no extreme-quant patches, no special CUDA setup

**Comparison to the prior 12B candidate I tried to download:**
I originally targeted a QAT-trained 12B from HuggingFace.
That file (7.38 GB, Q4_K_M) was hosted on HuggingFace's Xet CAS CDN,
which has a known bug that returns **corrupted GGUF headers** — the
`tensor_count` and `meta_count` fields come back as garbage integers.
I logged 10+ hours of partial downloads before confirming the headers
were broken. The fix was to switch to an official-model GGUF that
uses a standard blob backend instead of Xet. This is a real gotcha
for anyone downloading large community GGUFs from HuggingFace:
**always verify the GGUF header after download.**

```python
# Verify any GGUF before running it
import struct
with open("model.gguf", "rb") as f:
    magic = f.read(4)
    tensor_count = struct.unpack("<Q", f.read(8))[0]
    meta_count   = struct.unpack("<Q", f.read(8))[0]
assert magic == b"GGUF"
assert 100 < tensor_count < 5000
assert 1 < meta_count < 200
```

---

## The 8 GB reality: full offload is out

| Setting | What happens |
|---------|--------------|
| `-ngl 99` (full GPU) | ❌ OOM — 6.76 GB exceeds the ~6.3 GB VRAM death line |
| `-ngl 28` (mixed CPU+GPU) | ✅ Runs. KV cache fits on GPU, expert layers spill to CPU |
| `-ngl 0` (CPU only) | ✅ Runs but 3–5× slower than mixed |

**6.3 GB is the death line** on this card. This is the same limit
that killed the 22B video models and forced Bonsai-27B into
extreme Q1_0 quantization. Gemma-4-12B-Instruct is one of the
largest instruct-fine-tuned models that can fit on 8 GB at all,
and only with partial offload.

**VRAM used**: 844 MiB — plenty of headroom.
The 6.76 GB model barely registers against 8 GB of VRAM
because most of the weights live on CPU RAM.

---

## Speed: the partial-offload tax

Test conditions: `-n 64`, `-c 1024`, `-b 512`, `-t 8`, `-ngl 28`

| Mode | Prompt t/s | Generation t/s |
|------|-----------|----------------|
| ngl=28 (mixed) | 50.2 | **10.1** |
| Ability tests (average) | 78–106 | **9.7–10.0** |

**Comparison to prior models on this hardware:**

| Model | Quant | Offload | Generation t/s | Note |
|-------|-------|---------|----------------|------|
| Qwen3.6-35B-A3B | Q4_K_M | -ngl 99 + --n-cpu-moe | **30.79** | MoE dense on CPU, all attention on GPU |
| Gemma-4-12B-Instruct | Q4_K_S | -ngl 28 (mixed) | **10.1** | **This run** |
| Bonsai-27B | Q1_0 | -ngl 99 (prismml fork) | 34.5 | Extreme quant, hallucinates on facts |
| GLM-4.7-Flash | Q4_K_M | -ngl 28 (mixed) | ~8–10 (estimated) | 18 GB file, same mixed-offload class |

**Verdict**: Partial offload costs roughly **3× speed** vs a fully-GPU
model of the same size. For interactive chat (short answers, <200 tokens)
10 t/s is perfectly usable. For long-form generation (2,048 tokens)
it takes about **200 seconds** — comparable to a 7B model fully on GPU,
but with better per-token quality.

---

## Ability: 34/60 — a mixed bag

Six-dimension Chinese-language ability test. Each prompt scored 0–10
based on factual accuracy, completeness, and correctness of the output.

| Dimension | Score | Key observation |
|-----------|-------|-----------------|
| Code | 2/10 | Chain-of-thought block consumed the entire 512-token budget; the fibonacci function was never fully emitted |
| Translation | 7/10 | "投资自己是回报最高的，复利不只是钱的专利，知识和习惯同样适用" — natural, colloquial, accurate |
| Classical Chinese | 9/10 | Correctly identified 说 = 通假字 悦 (joy/delight); accurate translation of the 学而时习之 passage |
| Math | 10/10 | 鸡23兔12, full equation setup, correct arithmetic — textbook-correct |
| Logic (3-person paradox) | 4/10 | Case analysis started correctly (A≠Knight, A=Knave, B=Spy, C=Knight) but got truncated before reaching the final conclusion |
| Cultural knowledge | 2/10 | **Hallucination**: claimed 出师表 was written to 刘备 (Liu Bei). It was written by Zhuge Liang to 刘禅 (Liu Shan) during the 蜀汉 (Shu Han) period |
| **Total** | **34/60** | Mixed bag — strong on language, weak on reasoning under token budget |

### Highlights

**Strong suit: language quality.** Translation is natural and idiomatic.
Classical Chinese handling is surprisingly good — the 通假字 identification
for 说 = 悦 is a sign of real training depth, not pattern-matching.
Math is textbook-correct with proper equation setup.

**Weak suit: token budget discipline.** The model defaults to chain-of-thought
before every answer, which consumes ~250 tokens on a 512-token budget.
Code and logic tasks that need long reasoning chains get truncated
before reaching the actual answer. This is fixable in practice
by increasing `-n` to 1024 or using server mode with `enable_thinking: false`
(though the model does not reliably honor the prompt-level no-thinking
instruction — it kept emitting `[Start thinking]` blocks throughout).

**Hallucination flag: cultural knowledge.** The 出师表 answer is a
factual error that a 12B instruct model should not make. Zhuge Liang
wrote the 出师表 to Liu Shan (Emperor of Shu Han), not to Liu Bei
(who was already dead by the time of the Northern Expedition).
This is the same class of confident-hallucination error we saw
from Bonsai-27B Q1_0 — not surprising given the shared 12B-class
architecture and moderate quantization.

---

## How it was benchmarked

```bash
# Speed test (mixed offload — full GPU OOMs at 6.76 GB)
E:/WSL/llama.cpp/llama-cli.exe --model "E:/WSL/models/gemma-4-12b-it/gemma-4-12b-it-Q4_K_S.gguf" \
  -n 64 -c 1024 -b 512 -t 8 -ngl 28

# Ability tests (6 dimensions, -n 512, no-thinking prefix)
E:/WSL/llama.cpp/llama-cli.exe --model "..." -n 512 -c 2048 -b 512 -t 8 -ngl 28
# prompt: "不要使用 [Start thinking]，直接给出答案" + question + "\n/exit\n"
```

**Build**: ggml-org standard llama.cpp (b9946-fb30ba9a6).
Gemma-4-12B-Instruct is a standard GGUF and loads natively —
no prismml fork, no extreme-quant patches.

**Pagefile**: Windows 32 GB physical RAM + pagefile is required
(mmap-maps the entire 6.76 GB GGUF). On this host the
pagefile is 16–32 GB. Without it, `llama-cli.exe` crashes with
WinError 1455 on fork.

---

## Verdict: keep as a mid-range option

**Keep, but with clear boundaries.**

- **Use for**: Chinese translation, classical Chinese, math,
  creative writing, summarization where 7B is not enough.
- **Avoid for**: Code generation (CoT eats the token budget),
  long-form reasoning, and any task where factual accuracy
  on cultural/historical topics matters.
- **Server mode tip**: Run via `llama-server.exe` with
  `enable_thinking: false` — this avoids the CoT truncation
  problem and makes 12B usable for longer completions.
  Note: the model does not reliably respect prompt-level
  no-thinking instructions, so the server-side toggle is
  the only reliable way to suppress CoT.

**Where it sits in the 8 GB lineup:**

- **7B-class full GPU** → fastest, cheapest, good enough for
  simple tasks (15–30 t/s generation)
- **Gemma-4-12B-Instruct (this)** → mid-tier, better language
  quality, needs mixed offload (~10 t/s)
- **35B-A3B mixed offload** → best quality, MoE architecture
  lets it stay ~3× faster than a dense 35B would be (~30 t/s)

The 12B layer fills the gap between "7B is too dumb" and
"35B is too slow." It is not a主力, but it is a real option.

---

## Pitfalls and gotchas

1. **Xet CAS CDN corrupts GGUF headers** — if you download a
   community GGUF from HuggingFace and the header shows
   `tensor_count` in the trillions, the file is broken.
   Switch to an official-model GGUF or a different backend.
2. **-ngl 99 OOMs** — 6.76 GB exceeds the 6.3 GB VRAM
   death line. Always use mixed offload (`-ngl 20–30`).
3. **CoT truncation on `-n 512`** — use `-n 1024` for
   code and logic tasks, or disable thinking at the
   server level (`enable_thinking: false`).
4. **Prompt-level no-thinking is unreliable** — the model
   emits `[Start thinking]` blocks regardless of the
   system prompt. Use server mode, not CLI prompting.

---

## Series context

Part of the 8 GB local LLM series:

- v14.0 — Eight GB Pipeline Benchmark: Five Models, One GPU
- v15.0 — Bonsai-27B on 8 GB: How I Fit a 27B LLM on an RTX 5060
- **v19.0 — Gemma-4-12B-Instruct (this)**

The arc: from "what runs on 8 GB" to "what runs *well*".
This entry sits between the fully-GPU small models and
the mixed-offload giants. It is the first instruct-fine-tuned
12B in the series — not extreme quant, not MoE, just a
mainstream-size instruct model tested honestly.

---

*Run date: 2026-08-16*
*Model: google/gemma-4-12b-it (Q4_K_S, 6.76 GB, standard GGUF)*
*Hardware: RTX 5060 Laptop 8 GB / Ryzen 7 H 260 / 32 GB RAM / sm_120*
*Software: llama.cpp b9946-fb30ba9a6 (ggml-org standard build)*