# Roadmap

Every release ships a before/after number measured on real hardware (one headline figure + a bar chart, e.g.
"3x less RAM on the same chat"), so an upgrade is visible at a glance.

Guiding principle (borrowed from how the Elysia web framework cut memory): **don't load, don't copy, don't run what
isn't used** - skip unused model parts, share instead of duplicating caches, and specialise the launch per machine once.

## 0.1 - one command (released)
- `localllm`: detect GPU/RAM, pick the most accurate measured model for your language, download, start a tuned
  llama-server, open the chat page
- `localllm doctor`: which model sizes this GPU suits, speed estimates, how much text the chosen model holds
- `localllm eval`: Global-MMLU-Lite (23 languages) + INCLUDE (44 countries) + ThaiExam, logprob scoring
- Tuned launch: small-BAR fix, MTP drafting for Qwen3.8, single-slot unified KV, `-fit off`

## 0.2 - use less system RAM
- [x] `localllm chat`: terminal chat with streaming, history, /save, /think, Ctrl+C to stop (tested on gemma-4, 85-88 tok/s)
Evidence: llama-server defaults (`--cache-ram 8192` MiB prompt cache, `--ctx-checkpoints 32` per slot) took one Gemma 4
user from 0.7 GB to 18 GB of RAM and out-of-memory in three generations; with 0-1 checkpoints it stayed at 0.4-1.5 GB
(llama.cpp #21690, PR #16391).

- [ ] Measure RAM over a long chat, defaults vs tuned, Qwen3.8 + gemma-4 (#1)
- [ ] Low-RAM profile by default for one user: `-np 1`, `--ctx-checkpoints` 0-4 (0-1 for hybrid/Gemma 4),
      `--cache-ram` 0-1024 sized from installed RAM; check the speed cost (#2)
- [ ] KV cache `q8_0` by default (half the KV memory, ~0.05% perplexity); `q4_0` K only as an opt-in after measuring
      per language
- [ ] Load mode: read weights straight to VRAM when the model is fully offloaded; mmap only when experts stay in RAM
- [ ] Two-tier MoE estimate (VRAM + RAM) that warns when RAM is short - llama.cpp `--fit` assumes RAM is unlimited;
      choose `--n-cpu-moe` from free RAM (#4)
- [ ] `doctor`: RAM the model will use and what stays free; max context that fits on the GPU (#3)
- [ ] Don't load unused parts: skip the vision projector without images, skip the MTP head when not drafting
- [ ] Detect Windows "shared GPU memory" spill (VRAM silently overflowing into RAM) and say so

## 0.3 - work with the cloud providers' APIs
Evidence: llama-server already serves Anthropic `/v1/messages` (tools, vision, thinking) next to OpenAI
`/v1/chat/completions`; Ollama >= 0.14 does too; LiteLLM routes/falls back across providers; RouteLLM's router keeps 95%
of GPT-4 quality while sending only 26% of requests to it.

- [ ] One local endpoint: pass OpenAI and Anthropic APIs straight through to llama-server; thin shims for Ollama
      `/api/*` and Gemini `generateContent` (#5)
- [ ] Hybrid routing, local first: forward to the user's own cloud key when the prompt is too long, needs a tool/model
      the PC can't run, or the local model is busy; cost and privacy shown before sending; keys in the OS keychain (#6)
- [ ] Quality-aware routing: thresholds calibrated from *our measured per-language scores* (e.g. send hard Thai or math
      to the cloud when the local quant is below the quality floor) - nobody routes by language today
- [ ] `localllm route --test`: same prompt local vs cloud - answer, latency, cost (#7)

## 0.4 - squeeze the GPU
Evidence: ReBAR-off fix 1.7x on RX 9070 XT (ours) and 2.7x on RX 7900 XTX (#27097); MTP +40% on RDNA4 (ours), 1.86x on
RTX 3090, but slower on Apple Metal; CUDA fusion + `GGML_CUDA_GRAPH_OPT=1` +17-42% on RTX 4090/5090; Vulkan vs ROCm
winner on RDNA4 differs between decode and prefill.

- [ ] Squeeze step on first run: `llama-bench` every available backend (CUDA / HIP / Vulkan / SYCL) x flash attention x
      `-b/-ub`, cache the fastest per GPU + driver + model (#8)
- [ ] Detect a small host-visible heap (Resizable BAR off) and set the Vulkan fix automatically, confirmed by A/B
- [ ] `GGML_CUDA_GRAPH_OPT=1` on single-GPU NVIDIA
- [ ] MTP only where it measures faster: A/B draft length 2/3/5 per GPU, keep it on above 1.1x
- [ ] Pick the backend by workload: prefill-heavy (documents/RAG) vs decode-heavy (chat)
- [ ] Remove DeltaNet recurrent-state copy overhead (CPY/GET_ROWS of 3 MB states) and upstream it (#9)
- [ ] Faster load: 16 MiB upload staging buffers (1.2 s faster on a 12 GB model), skip the fit dry-run (#10)
- [ ] Before/after speed table per GPU in the release notes (#11)

## 0.5 - compression research (ongoing, results published per language)
Evidence: across 55 languages, 2-bit hurts non-Latin and low-resource languages most (Bengali -16 COMET vs ~-2 for
Japanese/French); language-specific imatrix helps only at 2-bit (~+3) and not at 4-bit (+-0.2) - matching our Thai null
result at 3.5 bpw. Below ~3 bits the weights are effectively restructured, so calibration alone can't fix it (ParetoQ).

- [ ] Quality floor: never recommend below UD-Q2_K_XL-class quants; warn below it
- [ ] Rank vendor QAT checkpoints (e.g. Gemma QAT) above post-training quants of the same model
- [ ] Per-language metric: KL divergence / top-1 agreement vs the BF16 model, not English perplexity
- [ ] Sensitivity-aware recipes: measure KLD per tensor, emit `--tensor-type` overrides, keep embeddings/output higher
      for non-Latin scripts
- [ ] Mixed multilingual chat-format imatrix (EN+TH+HI+AR+code+math) vs EN-only vs single-language, at 2-bit
- [ ] Better 2-bit formats on the multilingual set: IQ2_KT / IQ2_KL (ik_llama.cpp) and EXL3 ~2.5 bpw
- [ ] LoRA self-distillation of a 2-bit 27B from its Q8 teacher (no one has measured this per language yet)
- [ ] Vocabulary trimming per language for GGUF (no tool exists): smaller embedding/output and faster output layer,
      most useful on 1-4B models
- [ ] Publish every measured quant with its per-language scores on Hugging Face

## 0.6 - smart router: the right local model for each message (Laya-style)
Idea from the maintainer's Laya router (Local Router Chat): a small, fast classifier reads each message and sends it to
the model that is best *for that request* - fast MoE for everyday chat, the stronger dense model for Chinese/Japanese or
hard reasoning, the user's cloud key only when nothing local is good enough. Unlike generic routers, ours decides from
**measured per-language/per-task scores** in the catalog. Prior evidence: a few-shot LLM router picked correctly on 47/48
Thai/Thai-English requests vs 20/48 for rules, but took ~2.3 s per message; RouteLLM keeps 95% of GPT-4 quality with 26%
strong-model calls.

- [ ] Router latency budget < 50 ms: embedding/semantic router or a <=1B classifier on CPU, never the 27B model
- [ ] Two models resident at once (depends on 0.5 compression: e.g. a fast and a strong model that both fit in 16 GB),
      so routing never waits for a 5-6 s model swap; fall back to "stay on the current model" when a swap would be needed
- [ ] Routing table generated from catalog scores per language/task + measured tok/s, overridable per app
- [ ] Benchmark: answer quality and end-to-end latency vs a single model, on the multilingual suite + a routing test set
      (Thai/Thai-English set from the Laya research, extended to other languages); ship only if both improve
- [ ] Specialist pool for local-first routing: slots for code, math/reasoning, vision, embeddings (RAG + the router
      itself), speech-to-text (babelscribe), translation. A specialist joins the catalog only if it beats the generalist on
      its task by more than the benchmark margin (~5 points), fits next to the main model (or swaps fast), and isn't poor
      in the user's language (otherwise the generalist talks to the user and hands only the task to the specialist)
- [ ] `localllm eval` task suites beyond multiple choice: code (HumanEval+/LiveCodeBench-style), math (GSM8K/MATH-500),
      vision QA, translation - needed to measure specialists honestly
- [ ] Show which model answered and why, with a one-key override in `localllm chat`

## Later
- Shared prefix cache (block/radix, like vLLM/SGLang) instead of per-slot prompt copies - needs llama.cpp work
- More measured GPUs: `localllm eval` results from contributors feed the catalog
- Writing-quality evaluation (not just multiple choice)

Research behind this roadmap, with sources and numbers (Thai): [docs/research-th.md](docs/research-th.md).
