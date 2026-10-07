# make-localllm-easier — run the best local LLM your GPU can handle, in one command

**The leanest way to run local AI: least CPU, RAM and GPU memory, smallest files, measured quality.**

`localllm` picks, downloads and runs the most accurate local AI model for your PC and your language, chosen from real
benchmark measurements, with llama.cpp tuned for AMD, NVIDIA, Intel and Apple GPUs.

```
pip install make-localllm-easier
localllm
```

That's it. `localllm` checks your GPU and RAM, picks the most accurate model we have *measured* for your language that
fits your card, downloads llama.cpp and the model, starts it with settings profiled op by op, and opens the chat page.
You also get an OpenAI-compatible API at `http://127.0.0.1:8080/v1` for any app that speaks it. Offline, private, free.

```
localllm chat       # chat right here in the terminal (Thai, Japanese, any language)
localllm doctor     # what this GPU is good for: model sizes, speed, how much text it can hold
localllm list       # every model we have measured, with scores per language
localllm serve      # API only, no browser
localllm eval       # score any running server in English + your language
```

## FAQ

**Which local LLM should I run on my GPU?** Run `localllm doctor`. It lists which model sizes fit your card (4B up to
120B MoE), at which quantization, how fast they should run, the most accurate measured model for your language, and
how much system RAM that model uses (est.).

**Can a 16 GB GPU run a 27B model?** Yes. Qwen3.8-27B at ~3.5 bits (12.2 GB) runs at ~50 tok/s on an RX 9070 XT and
keeps 81.8% on English Global-MMLU-Lite. gemma-4-26B-A4B (13.3 GB) runs at ~69 tok/s with similar accuracy.

**Is a 2-bit quantized model good enough?** Usually not for non-English use: 2-bit costs 8-13 accuracy points, and
Hindi, Arabic and Thai lose the most (13 points).

**Why is llama.cpp slow on my AMD (or Intel) GPU on Windows?** If Resizable BAR is off, llama.cpp's Vulkan backend puts
buffers in a 256 MB host-visible heap backed by system RAM and decode drops up to 1.7x. `localllm` sets
`GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` for you ([llama.cpp#27097](https://github.com/ggml-org/llama.cpp/issues/27097)).

**Can I chat with a local LLM in the terminal?** Yes: `localllm chat`. Answers stream as they're written, the
conversation is remembered, `/save` writes it to a file, `/think` shows the model's reasoning, Ctrl+C stops an answer.

**Does it work offline?** After the first download, yes. Nothing leaves your PC.

**Which languages are measured?** 23 languages on Global-MMLU-Lite, 44 countries' own exams on INCLUDE, plus Thai
(ThaiExam). `localllm eval --langs ...` measures any of them on your hardware.

## What `localllm doctor` tells you

```
GPU AMD Radeon RX 9070 XT  15.9 GB  (640 GB/s)    RAM 32 GB    language: th

Model sizes for this PC (whole model on the GPU = fast):
  [OK  ] 4B                       Q8 4.2 GB  ~90 tok/s (est.)
  [OK  ] 8B                       Q8 8.5 GB  ~45 tok/s (est.)
  [OK  ] 14B                      Q6 11.5 GB  ~33 tok/s (est.)
  [OK  ] 24-32B                   Q3 13.2 GB  ~29 tok/s (est.)
  [SLOW] 30B MoE (3B active)      Q4 18.0 GB with experts in RAM - works, ~10-25 tok/s
  [NO  ] 70B                      needs ~42.0 GB - too big for this PC

Best measured model for you: gemma4-26b-a4b-qat  (MoE with ~4B active params: fastest)
What it can do here:
  TH  real local school/licence exams   65.7% correct  <- your language
  holds ~78k tokens at once (~130 pages of text) next to the model
  uses ~34.7 GB of system RAM: ~0.3 GB embeddings/CPU-mapped + ~8.0 GB prompt cache + ~25.9 GB ctx checkpoints + ~0.5 GB host (est.)
  leaves ~0 GB of RAM free for other apps (est.)
  answers at ~69 tok/s
```

Speeds marked *est.* come from your card's memory bandwidth, calibrated on measured runs. Everything else is measured.

## Measured results (RX 9070 XT 16 GB, Windows 11, llama.cpp Vulkan)

Accuracy (%) on multiple-choice exams, zero-shot. **global** = Global-MMLU-Lite: the same 400 questions translated, so
languages compare like for like. **regional** = INCLUDE: real exams written in each country (ThaiExam for Thai).

| | Qwen3.8-27B Q3 (12.2 GB) | gemma-4-26B-A4B QAT Q4 (13.3 GB) | Qwen3.8-27B 2-bit (7.8 GB) |
|---|---|---|---|
| English | 81.8 | 82.2 | 74.2 |
| Chinese | 76.2 / 74.7 | 73.5 / 66.5 | 67.8 / 67.8 |
| Spanish | 80.2 / 76.8 | 74.5 / 75.2 | 70.8 / 69.2 |
| Japanese | 73.5 / 87.6 | 74.5 / 81.9 | 65.8 / 77.9 |
| Arabic | 70.8 / 71.2 | 71.5 / 73.6 | 60.8 / 57.2 |
| Hindi | 69.0 / 74.3 | 69.5 / 71.0 | 56.2 / 55.5 |
| Thai | – / 67.1 | – / 65.7 | – / 54.2 |
| **decode speed** | **50 tok/s** (MTP) | **69 tok/s** | 40 tok/s |

Cells are global / regional. Margins are about ±4 (global) and ±5 (regional) points at 95%, so `localllm` treats gaps
under 2 points as a tie and picks the faster model.

## Findings worth knowing

1. **2-bit costs 8-13 points, and lower-resource languages pay the most.** Hindi, Arabic and Thai lose 13; English,
   Chinese and Spanish about 8-9. A 177B MoE squeezed to 1.6 bits scored *below* a 27B at 3 bits.
2. **Calibrating the quantization on your language doesn't help at ~3.5 bits.** A Thai-text importance matrix scored the
   same as the stock one in Thai, English and Chinese (64.8 vs 64.6 Thai). At this level the number of bits matters,
   the calibration text doesn't.
3. **AMD/Intel cards without Resizable BAR lose up to 1.7x decode speed** in llama.cpp's Vulkan backend. Hybrid DeltaNet
   models (Qwen3.5/3.8) suffer most: they rewrite a 3 MB state per layer per token.
4. **Qwen3.8 GGUFs ship a multi-token-prediction head.** Drafting 2 tokens with it adds ~40% decode speed for free;
   drafting 3 is slower.
5. **The first run of a new llama.cpp build is slow** while the GPU driver compiles its shaders once (~15 s).

## How the benchmark works

`localllm eval` asks each question with thinking off and reads the log-probability of every answer letter from the first
generated token, then takes the most likely one. It's prompt processing only, so a language takes a few minutes, and the
result is deterministic. Data is downloaded at eval time from the original Apache-2.0 datasets
([Global-MMLU-Lite](https://huggingface.co/datasets/CohereLabs/Global-MMLU-Lite),
[INCLUDE](https://huggingface.co/datasets/CohereLabs/include-lite-44),
[ThaiExam](https://huggingface.co/datasets/typhoon-ai/thai_exam)) and never redistributed. It measures knowledge and
reasoning in multiple choice, not writing quality.

## Contributing

The catalog only grows with measurements. Run `localllm eval --langs en,<yours>` on your GPU and open a PR with
`~/.localllm/results.json` and your GPU name. Other languages' local exams are very welcome. See [ROADMAP.md](ROADMAP.md)
for what's next: using less system RAM (0.2), working alongside cloud provider APIs (0.3), a speed-only release (0.4), and per-language compression research (0.5).

## License

MIT. Models keep their own licenses; benchmark data keeps its own (Apache-2.0).
