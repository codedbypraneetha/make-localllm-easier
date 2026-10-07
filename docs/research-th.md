# วัดคุณภาพรายภาษา คือช่องว่างที่ยังไม่มีใครเติม

**สรุปก่อน (BLUF):** งานวิจัยที่มีอยู่บอกตรงกันว่า **"วัดคุณภาพจริงต่อ quant และต่อภาษา" คือจุดต่างที่ make-localllm-easier ยังครองได้คนเดียว** เครื่องมือที่ใกล้เคียงที่สุดคือ llmfit, llama.cpp `--fit`, Ollama scheduler, LM Studio estimator และ HF hardware filter แต่ทุกตัวตอบแค่ "ลงได้ไหม / กี่ layer อยู่บน GPU" หรือใช้คะแนนคุณภาพแบบ proxy ที่ไม่มีมิติภาษา ผลที่คุณวัดเองบน RX 9070 XT สอดคล้องกับวรรณกรรมทุกจุด
- 2-bit เสีย 8-13 จุด โดย Hindi/Arabic/Thai เสียมากสุด ตรงกับรูปแบบ "สคริปต์ non-Latin และภาษา low-resource เสียหนักกว่า"
- imatrix ภาษาไทยไม่ช่วยที่ ~3.5 bpw ตรงกับผลที่ calibration ภาษาเดียวกันต่างกันแค่ ±0.2 COMET ที่ 4-bit และช่วยเพียง ~+3 ที่ 2-bit

ข้อสรุปเชิงปฏิบัติจึงเป็นแบบนี้
- **calibration อย่างเดียวปิดช่องว่าง 2-bit ไม่ได้** คันโยกที่ใหญ่กว่าคือ dynamic mixed precision, codec แบบ trellis/codebook และ QAT/distillation
- **ด้าน RAM** ศัตรูอันดับหนึ่งคือค่า default ของ llama-server เอง: `--cache-ram 8192` และ `--ctx-checkpoints 32` ไม่ใช่ตัวน้ำหนักโมเดล
- **ด้านความเร็ว** MTP กับการเลือก backend/ตั้งค่าเครื่อง (ReBAR, CUDA graph) ให้ผลมากกว่าการเปลี่ยน engine
- **ด้าน API** llama-server รองรับ Anthropic `/v1/messages` แบบ native แล้ว งาน 0.3 จึงเหลือแค่ shim และ router

**ป้ายความพร้อมที่ใช้ในรายงานนี้**
- `[paper]` = งานวิจัยตีพิมพ์
- `[research code]` = มีโค้ด แต่ยังไม่ใช่ของสำหรับผู้ใช้ทั่วไป
- `[usable tool]` = ใช้ได้จริงวันนี้
- **(แหล่งรอง)** = บล็อก/สรุปจากบุคคลที่สาม หรือหน้าที่ยืนยันไม่ได้ ให้ถือว่าความน่าเชื่อถือต่ำกว่า

ตัวเลขทุกตัวในรายงานมาจากแหล่งที่ลิงก์ไว้ ยกเว้นตัวที่ระบุว่าเป็น "การคำนวณ/อนุมานของผู้วิจัย"

## Compression: 2-bit พังเพราะน้ำหนักถูก "ปรับโครงสร้าง" ไม่ใช่เพราะ calibration ไม่ดี

**Mixed precision คือคันโยกที่สุกงอมที่สุดใน GGUF** `[usable tool]`
- Unsloth Dynamic เลือกชนิด quant แยกตาม layer
- Dynamic 3.0 (2026) ใช้ชุด calibration ใหม่ ">1.5M tokens" ที่เน้น multilingual/agentic ([Unsloth Dynamic 3.0](https://unsloth.ai/docs/basics/dynamic-3.0-ggufs.md))
- Unsloth ระบุว่า "calibration ด้วย plain text ไม่ได้ผลกับ instruct model ที่มี chat template"
- Dynamic 3.0 ตัดโมดูล MTP ออกจาก quant ที่ต่ำกว่า UD-Q2_K_XL ซึ่งประหยัด ~500 MB
- Gemma 3 27B: **Q2_K_XL ได้ 68.70% MMLU เทียบกับ Google QAT 67.77%** และ Q4_K_XL ได้ 71.47% เทียบกับ QAT 71.07% ที่ 15.64 GB

ตัวเลขเหล่านี้เป็น**ตัวเลขของผู้ขายที่วัดด้วย harness ของตัวเอง** Unsloth เองก็บอกว่า reproduce MMLU ทางการของบางโมเดลไม่ได้ ([Unsloth Dynamic 2.0](https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs), แหล่งรอง: snippet จากการค้นหา เพราะหน้าจริงเปิดไม่ได้)

สำหรับ Qwen3.8-27B Unsloth อ้างว่า
- UD-Q2_K_XL (9.83 GB) ดีกว่าเจ้าอื่น +8% top-1
- **ต่ำกว่า ~2 bpw คือหน้าผา** UD-IQ1_S ทำให้เกิด "excessive looping", ตอบว่าง และ tool-calling พัง

ผลที่คุณวัดได้ (−8 ถึง −13 จุดที่ 2-bit) จึงเป็นเหตุผลให้ตั้ง **floor เริ่มต้นไว้ที่ "UD-Q2_K_XL ขึ้นไป"**

**Trellis/codebook ชนะ IQ2/IQ3 ที่ bpw เท่ากัน**
- ik_llama.cpp IQ2_KT (2.125 bpw) `[usable tool, แต่เป็น fork]`
  - ใช้บิตน้อยกว่า IQ2_XS ~0.2 bpw ที่ error เท่ากัน
  - error 28.7% เทียบกับ QTIP 33.2% บน Llama-2-7B
  - decode บน RTX 4080 ได้ 194 t/s เทียบกับ IQ2_XS 208-216 t/s
  - quantize โมเดล 8B บน CPU ใช้เวลาแค่ 2.5-4.5 นาที ([ik_llama.cpp PR #113](https://github.com/ikawrakow/ik_llama.cpp/pull/113))
  - ข้อควรระวัง: type เหล่านี้ (IQ*_KT, IQ*_K, _R4) **น่าจะโหลดใน mainline llama.cpp ไม่ได้** (อนุมาน) และยังไม่พบหลักฐานว่ารองรับ AMD/Vulkan ซึ่งสำคัญมากสำหรับเครื่อง RX 9070 XT ของคุณ
- QTIP `[paper + research code, NeurIPS 2024]` decode 2-bit Llama-2-7B ได้ 188 t/s เทียบกับ QuIP# 186 และ AQLM 81.5 บน RTX 6000 Ada ([QTIP GitHub](https://github.com/Cornell-RelaxML/qtip), [arXiv 2406.11235](https://arxiv.org/abs/2406.11235))
- EXL3 (ExLlamaV3, ต่อยอดจาก QTIP) `[usable tool, NVIDIA]` เทียบกับ GGUF บน RTX 5090 ([Kaitchup](https://kaitchup.substack.com/p/serving-exllamav3-with-tabbyapi-accuracy), แหล่งรอง)
  - คุณภาพต่อ GB ดีกว่า
  - prefill เร็วกว่าถึง 2.5x
  - แต่ llama.cpp decode เร็วกว่า 15-21%
- AQLM `[research code]` ให้คุณภาพ 2-bit ดี เช่น Llama-3-70B MMLU 0.79 → 0.75 แต่ quantize แพงและ decode ช้า ([AQLM GitHub](https://github.com/Vahe1994/AQLM))
- BCJR-QAT (2026) `[paper]` ทำ QAT สำหรับ trellis code ที่ deploy ผ่าน IQ2_KT ได้ตรง 2 bpw ([arXiv 2605.10655](https://arxiv.org/pdf/2605.10655))

**QAT ให้ผลจริง แต่เป็นงานของผู้ทำโมเดล**
- Gemma 3 QAT ลด perplexity drop ที่ Q4_0 ได้ **54%** และ 27B ลดจาก 54 → 14.1 GB `[usable tool]` ([Google Developers Blog](https://developers.googleblog.com/en/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/))
- ParetoQ `[paper, NeurIPS 2025]` ([arXiv 2502.02631](https://arxiv.org/abs/2502.02631))
  - พบ **"learning transition" ระหว่าง 2 กับ 3 บิต** ที่ ≥3 บิตน้ำหนักแทบไม่ขยับจาก pretrained แต่ที่ ≤2 บิตน้ำหนักเปลี่ยนโครงสร้างมาก
  - นี่คือกลไกที่อธิบายผลของคุณได้ (อนุมาน): PTQ ~3.5 bpw ไม่เป็นอันตราย ส่วน 2-bit ต้องซ่อมด้วย training ไม่ใช่ calibration
- BitNet b1.58 2B4T ใช้หน่วยความจำ non-embedding 0.4 GB, MMLU 53.2, CPU 29 ms/token `[usable tool via bitnet.cpp]` ([InfoQ](https://www.infoq.com/news/2025/04/microsoft-bitnet-1bit-llm))
  - ยังไม่มีหลักฐานว่า BitNet รักษาคุณภาพ multilingual ได้

**Pruning/distillation และ vocabulary trimming**
- Minitron `[paper + checkpoints]` ([arXiv 2407.14679](https://arxiv.org/abs/2407.14679))
  - ย่อ Nemotron-4 15B → 8B/4B ด้วยข้อมูลเทรนไม่ถึง 3% ของเดิม
  - MMLU ดีกว่าการเทรนจากศูนย์สูงสุด 16%
- การตัด layer ลึก `[paper]` ตัดได้ถึงครึ่งหนึ่งแล้วซ่อมด้วย QLoRA แต่ QA เสียหนัก ([arXiv 2403.17887](https://arxiv.org/abs/2403.17887))
- Vocabulary trimming `[paper + PyPI vocabtrimmer]` ([arXiv 2305.15020](https://arxiv.org/pdf/2305.15020), [vocabtrimmer](https://pypi.org/project/vocabtrimmer/0.0.1))
  - ใช้ vocab ~50% ก็รักษาคุณภาพได้ และ embedding อาจเกิน 80% ของพารามิเตอร์ใน mBART/mT5
  - แต่สำหรับ decoder 27B ที่มี vocab ~150k ส่วน embedding+output เป็นแค่ไม่กี่เปอร์เซ็นต์ (ประมาณการ ไม่มีแหล่ง) จึงคุ้มกับโมเดล 1-4B มากกว่า
  - **ยังไม่มีเครื่องมือ trim GGUF**
  - ไม่มีงานไหนวัดผลของ pruning แยกตามภาษาไทย/ฮินดี/อาหรับ

**Multilingual: ภาษาที่ baseline แย่ จะเสียหนักกว่า**
- Marchisio et al. `[paper, EMNLP Findings 2024]` ([arXiv 2407.03211](https://arxiv.org/pdf/2407.03211))
  - สคริปต์ non-Latin เสียมากกว่า และ math เสื่อมเร็วสุด
  - **automatic metric ประเมินความเสียหายต่ำเกินจริง**: ญี่ปุ่นลด 1.7% ตาม metric อัตโนมัติ แต่ลด 16.0% ตามคนประเมิน
- Marie & Fujita `[paper, 2025]` (55 ภาษา, [arXiv 2508.20893](https://arxiv.org/html/2508.20893v1))
  - Qwen3-8B ที่ Q2_K แปล EN→Bengali ได้ 76.1 → 59.9 ขณะที่ญี่ปุ่น/ฝรั่งเศสเสียแค่ ~2 จุด
  - GGUF+imatrix ทนที่สุดแม้ที่ 2-bit
  - imatrix ภาษาเบงกาลีให้ +3.1 COMET ที่ 2-bit และ ±0.2 ที่ 4-bit
- "Calibrating Beyond English" `[paper, 2026]` ([arXiv 2601.18306](https://arxiv.org/html/2601.18306v1))
  - **calibration แบบผสมหลายภาษา** ลด PPL ของ GPTQ ได้ 3.52 และเพิ่ม Global-MMLU +1.55
  - แต่ AWQ ไวต่อเรื่องนี้น้อยมาก
  - ภาษาใกล้กันช่วยกันได้

ช่องว่างที่ชัดเจน: **ยังไม่มีงานไหนวัดภาษาไทยโดยตรง** และยังไม่มีงานที่รวม language calibration เข้ากับ trellis quantizer

## Model sizing: ระบบ fit ถูกแก้ไปแล้ว แต่ระบบ rank ตามคุณภาพยังว่าง

| เครื่องมือ | ทำอะไร | ใช้ข้อมูลคุณภาพ? | ป้าย |
|---|---|---|---|
| llmfit ([GitHub](https://github.com/AlexsJones/llmfit), [how-it-works](https://github.com/AlexsJones/llmfit/blob/main/docs/how-it-works.md)) | ให้คะแนน quality/speed/fit/context โดยประมาณ tok/s จาก `bandwidth/size × 0.55` ~37.7k stars | proxy: ขนาดพารามิเตอร์ + ชื่อเสียง family + leaderboard **ไม่มีมิติภาษา** | `[usable tool]` คู่แข่งตรงที่สุด |
| llama.cpp `--fit` / `llama-fit-params` ([README mirror](https://huggingface.co/datasets/echodict/llama.cpp/blob/main/tools/fit-params/README.md)) | ตั้ง `-ngl`, tensor split และ `-ot` ของ MoE ให้พอดี VRAM ว่าง และลด context ได้ **โดยถือว่า RAM ไม่จำกัด** | ไม่ใช้ | `[usable tool]` default on |
| Ollama scheduler ([blog](https://ollama.com/blog/new-model-scheduling)) | เปลี่ยนจากประมาณการเป็น**วัดจริง**ก่อนโหลด เช่น gemma3:12b 128k ได้ 52 → 85.5 tok/s บน 4090 | ไม่ใช้ | `[usable tool]` |
| LM Studio `lms load --estimate-only` ([docs](https://www.lmstudio.ai/docs/cli/local-models/load)) | ประมาณ GPU/total memory รวม context, FA และ vision | ไม่ใช้ และไม่เทียบตัวเลือก | `[usable tool]` |
| HF hardware filter ([yuv.ai](https://yuv.ai/blog/huggingface-hardware-filter), **แหล่งรอง**) | เทียบขนาดไฟล์ GGUF กับ RAM/VRAM ที่ผู้ใช้ประกาศ ไม่ระบุว่าคิด KV ด้วย | ไม่ใช้ | `[usable tool]` |
| gpu_poor ([GitHub](https://github.com/rahulschand/gpu_poor)) | web calculator ที่อ้างว่าแม่นยำ "within 500MB" | ไม่ใช้ | `[usable tool]` |
| Intel Low-Bit Leaderboard ([Intel](https://www.intel.com/content/www/us/en/developer/articles/technical/low-bit-quantized-open-llm-leaderboard.html)) | ความแม่นยำของโมเดลที่ quantize แล้ว (AutoRound/GPTQ/AWQ/GGUF) | ใช่ แต่ไม่ hardware-aware และไม่แยกภาษา | `[usable tool]` (ยังไม่ยืนยันว่ายังอัปเดตอยู่) |
| LocalScore ([GitHub](https://github.com/cjpais/LocalScore)), llama.cpp scoreboards ([CUDA #15013](https://github.com/ggml-org/llama.cpp/discussions/15013), [Vulkan #10879](https://github.com/ggml-org/llama.cpp/discussions/10879)) | ฐานข้อมูลความเร็วต่อ GPU แบบ crowdsourced | วัดแค่ความเร็ว | `[usable tool]` |

ข้อสรุปเชิงกลยุทธ์ (อนุมาน)
- **อย่าเขียน memory placement ใหม่** ให้เรียก `llama-fit-params` ในขั้นวางโมเดลสุดท้าย แล้วเก็บ estimator ของตัวเองไว้ใช้ "rank ก่อนดาวน์โหลด"
- จุดต่างที่ทำได้มีสามข้อ
  1. คะแนนวัดจริงต่อ quant และต่อภาษา เป็น Pareto ระหว่าง "คุณภาพที่เสีย" กับ "ความเร็วที่ได้"
  2. แสดง **context สูงสุดที่รันบน GPU ได้เป็น output** แทนที่จะให้ผู้ใช้กรอก context เป็น input
  3. ตรวจฝั่ง RAM ของ MoE offload ซึ่ง `--fit` ถือว่าไม่จำกัด
- calibrate ค่าคงที่ 0.55 ต่อ GPU จาก tg128 ใน scoreboard ได้ เช่น 5090 ได้ 290 t/s × ~3.8 GB ≈ 1.1 TB/s คิดเป็น ~0.6 ของ peak (ตัวเลขจาก [knightli](https://knightli.com/en/2026/04/23/llama-cpp-gpu-benchmark-cuda-rocm-vulkan-scoreboard/) ซึ่งเป็น**แหล่งรอง** ส่วนตัวคูณเป็นการคำนวณของผู้วิจัย)

## RAM/VRAM: ค่า default ของ llama-server กิน RAM มากกว่าตัวโมเดล

**KV-cache quantization**
- llama.cpp `[usable tool]` รองรับ `--cache-type-k/-v` ได้แก่ q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1 ([llama-server man page](https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html)) แต่การ quantize V ต้องเปิด flash attention
- ผลต่อคุณภาพ ([techplained](https://www.techplained.com/kv-cache-quantization), **แหล่งรอง ทดสอบโมเดลเดียว**)
  - q8/q8 เพิ่ม PPL แค่ ~+0.05%
  - q4/q4 เพิ่ม ~+2.1%
  - **ยังไม่มีใครวัดผลของ KV quant แยกตามภาษา** ซึ่งเป็นช่องที่ launcher ทดสอบเองได้
- งานวิจัยที่ประหยัดได้มากกว่า แต่**ไม่มีใน llama.cpp**
  - KIVI 2-bit ใช้ peak memory น้อยลง 2.6x `[paper + research code]` ([arXiv 2402.02750](https://arxiv.org/abs/2402.02750v2))
  - KVQuant 3-bit เพิ่ม PPL ไม่ถึง 0.1 `[paper]` ([arXiv 2401.18079](https://arxiv.org/abs/2401.18079))
  - SnapKV memory efficiency ดีขึ้น 8.2x ที่ 16K `[paper]` ([arXiv 2404.14469](https://arxiv.org/abs/2404.14469))

**Prompt cache และ checkpoints คือหลุมใหญ่ที่สุด** `[usable tool, แต่ default อันตราย]`
- default คือ `--cache-ram 8192` MiB และ `--ctx-checkpoints 32` ต่อ slot ([man page](https://manpages.debian.org/unstable/llama.cpp-tools/llama-server.1.en.html))
- ใช้งานจริงที่ default เห็น RAM ~10.8 GB ([PR #16391](https://github.com/ggml-org/llama.cpp/pull/16391))
- Issue #21690 (Gemma 4) RAM ขึ้นเป็น **0.7 → 10 → 18 GB → OOM ภายใน 3 generation**
  - ตั้ง `--ctx-checkpoints 1 -np 1` แล้วนิ่งที่ 1.5 GB
  - ตั้ง `0` แล้วนิ่งที่ 0.4 GB
  - issue ถูกปิดแบบ "not planned" ([issue #21690](https://github.com/ggml-org/llama.cpp/issues/21690))
- เทียบกับ vLLM ที่ hash KV block แล้วแชร์ prefix ซ้ำกันพร้อม LRU ภายใน KV pool เดียว ([vLLM docs](https://docs.vllm.ai/en/latest/design/prefix_caching.html)) ส่วน llama.cpp เก็บ state ทั้ง prompt เป็นก้อน **จึงน่าจะเก็บ system prompt ที่ซ้ำกันสองครั้ง** (อนุมาน ยังไม่ยืนยัน)

**Hybrid model (Gated DeltaNet)**
- KV ที่โตตาม context ถูกแทนด้วย state ขนาดคงที่ ([arXiv 2603.05931](https://arxiv.org/html/2603.05931v1))
- คำนวณจาก config ของ Qwen3-Next-80B-A3B ([config.json](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct/raw/main/config.json)) ได้ประมาณนี้ (**การคำนวณของผู้วิจัย ยังไม่ได้วัด**)
  - state ~75 MiB ต่อ sequence
  - KV ~24 KiB/token เทียบกับ ~128 KiB ของ dense 8B
  - **แต่ checkpoint 32 อัน × ~75 MiB ≈ 2.4 GB ต่อ slot**
- สำหรับโมเดลตระกูลนี้ `--ctx-checkpoints` จึงเป็นปุ่ม RAM หลัก

**Load mode และ offload**
- `--load-mode` มีตัวเลือก auto, mmap, mlock, mmap+mlock และ dio `[usable tool]` ถ้าโมเดลอยู่บน GPU ทั้งหมด ใช้ `--no-mmap` หรือ dio เพื่อไม่ให้ page cache ของไฟล์ค้างอยู่ใน RAM (อนุมาน)
- `--n-cpu-moe` / `-ot` `[usable tool]` ([discussion #15396](https://github.com/ggml-org/llama.cpp/discussions/15396))
  - gpt-oss-20b บน RTX 3060 12 GB ได้ 64 tok/s ที่ 16K (`-ncmoe 2`) เทียบกับ 75 tok/s เมื่ออยู่บน GPU ทั้งหมด
  - discussion เดียวกันเตือนว่า Windows driver อาจ spill VRAM ไปยัง shared memory แบบเงียบๆ
- งานวิจัย offload อื่นๆ
  - PowerInfer เร็วขึ้นสูงสุด 11x แต่ใช้ได้เฉพาะโมเดล ReLU-sparse `[research code, ใช้งานได้]` ([GitHub](https://github.com/SJTU-IPADS/PowerInfer))
  - Fiddler, HOBBIT และ MoE-Infinity `[paper/research code]` วัดเทียบ baseline เก่าหรือบน Jetson ([2402.07033](https://arxiv.org/abs/2402.07033), [2411.01433](https://arxiv.org/pdf/2411.01433), [2401.14361](https://arxiv.org/html/2401.14361v3)) และแนวคิดหลักของ Fiddler (คำนวณ expert บน CPU) ก็อยู่ใน llama.cpp แล้ว
  - Gemma 3n PLE ย้าย per-layer embedding ออกจากหน่วยความจำ accelerator ได้ ทำให้ E2B ที่มีพารามิเตอร์ดิบ >5B เหลือ ~1.91B effective `[usable tool ใน Google stack]` ([Gemma 3n docs](https://ai.google.dev/gemma/docs/gemma-3n)) ยังไม่ได้ยืนยันว่า llama.cpp ทำแบบเดียวกันหรือไม่

## GPU speed และ interop: MTP กับการตั้งค่าเครื่องชนะการเปลี่ยน engine

**MTP** `[usable tool, ใหม่มาก]`
- MTP เข้า llama.cpp ผ่าน PR #22673 ใช้ `--spec-type draft-mtp --spec-draft-n-max N` โดย weight อยู่ใน GGUF เดียวกัน ([mer.vin](https://mer.vin/2026/05/run-qwen-3-6-mtp-in-llama-cpp-faster-local-inference-with-built-in-speculative-decoding/), **แหล่งรอง**)
- ผลที่รายงาน
  - Qwen 3.6 27B บน RTX 3090 เร็วขึ้น ~1.86x (**แหล่งรอง**)
  - Strix Halo เร็วขึ้น 3.0x ที่ Q8/q8 และ "4.8x" ซึ่ง**เปลี่ยน quant และ KV type ไปด้วย จึงไม่ใช่ผลของ MTP ล้วน** ([sleepingrobots](http://sleepingrobots.com/dreams/mtp-qwen36-strix-halo/), **แหล่งรอง**)
  - บน Apple Metal **ช้าลงสูงสุด 28%**
- n-gram และ draft model ข้ามโมเดลวัดได้ −0.3% / −0.2% บน Strix Halo
- ผลของคุณ (+40% ที่ draft 2 token บน RX 9070 XT) อยู่กลางช่วงนี้ และเป็น**ตัวเลข RDNA4 dGPU ตัวแรกที่มี** เพราะในโน้ตไม่พบตัวเลข MTP ของ RDNA4 จากที่อื่น
- EAGLE-3 acceptance 0.6-0.8 แต่ใช้ใน vLLM/SGLang ไม่ใช่ llama.cpp `[paper + datacenter tool]` ([LMSYS](https://lmsys.org/blog/2025-12-01-eagle3-vertex/))

**Backend และ kernel fusion**
- CUDA fusion + `GGML_CUDA_GRAPH_OPT=1` ให้ tg128 เพิ่ม **+17% ถึง +42%** เช่น 5090 Qwen3-30B-A3B ได้ 247 → 352 t/s `[usable tool]` ([am17an](https://am17an.bearblog.dev/new-post/), บล็อกของผู้พัฒนาเอง)
- RX 9070 XT (**แหล่งรอง**, [glukhov](https://glukhov.org/llm-hosting/comparisons/amd-rocm-vs-vulkan-llm-hosting/))
  - Vulkan tg เร็วกว่า HIP 13-30% แต่ prefill ที่ prompt ยาว HIP อาจชนะ ([OpenBenchmarking](https://openbenchmarking.org/result/2509078-NE-ROCMVSVUL92))
  - ข้อมูลขัดกันเองและเปลี่ยนตาม build
  - รายงาน regression ที่ Vulkan gfx1201 ช้ากว่า HIP 4.7-6.7x เมื่อ hidden size ≥4096 **ยังไม่มีเลข issue ยืนยัน**
- **ReBAR** issue #27097 (RX 7900 XTX, Linux) ([llama.cpp #27097](https://github.com/ggml-org/llama.cpp/issues/27097))
  - ปิด ReBAR ได้ tg128 13.89 t/s เปิด `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` แล้วได้ 37.09 t/s (2.7x)
  - issue ยังเป็น bug-unconfirmed
  - ผล 1.7x ของคุณจึงเป็นหลักฐานยืนยันอิสระชิ้นที่สอง
- ik_llama.cpp ช้ากว่ามากใน prefill กับ UD-Q4_K_XL (14 เทียบกับ 125 t/s บน 3090) แต่สูสีบน IQ4_NL ([hardware-corner](https://www.hardware-corner.net/guides/llama-cpp-vs-ik_llama-cpp/), **แหล่งรอง**)

**Interop และ hybrid routing**
- llama-server มี `POST /v1/messages` และ `/count_tokens` รองรับ tools, vision และ thinking ใช้กับ Claude Code ได้ด้วย `ANTHROPIC_BASE_URL` `[usable tool]` ([HF blog](https://huggingface.co/blog/ggml-org/anthropic-messages-api-in-llamacpp), [PR #17570](https://github.com/ggml-org/llama.cpp/pull/17570))
- Ollama รองรับเหมือนกันตั้งแต่ v0.14.0 ([docs](https://docs.ollama.com/api/anthropic-compatibility))
- LiteLLM รับ `/v1/messages` แล้ว route ต่อไป OpenAI/Gemini/Ollama ได้ พร้อม fallback และ cost tracking ([LiteLLM](https://docs.litellm.ai/docs/anthropic_unified/))
- RouteLLM-MF `[paper + research code]` ([LMSYS](https://lmsys.org/blog/2024-07-01-routellm/), [arXiv 2406.18665](https://arxiv.org/abs/2406.18665))
  - ได้ 95% ของคุณภาพ GPT-4 บน MT Bench โดยเรียก GPT-4 แค่ 26% (หรือ 14% เมื่อใช้ augmented data)
  - threshold ถูกจูนกับคู่ GPT-4/Mixtral จึงต้อง recalibrate ให้เข้ากับโมเดล local ของผู้ใช้
- ยังไม่ยืนยันว่า llama-server รองรับ Gemini `generateContent` หรือ Ollama `/api/chat` แบบ native

## สิ่งที่ควรใส่ใน roadmap

| เวอร์ชัน | ไอเดีย (เรียงตามความคุ้ม) | หลักฐาน |
|---|---|---|
| **0.2 RAM น้อยลง** | (1) โปรไฟล์ "low-RAM" เป็น default สำหรับ single-user: `-np 1`, `--ctx-checkpoints 1-4` (hybrid/Gemma 4 ใช้ 0-1), `--cache-ram 0-1024` (2) `-fa on` + KV `q8_0/q8_0` เป็นค่าเริ่มต้น และเปิด q4_0 ได้เฉพาะ K ในโหมด aggressive หลังวัดผลรายภาษา (3) `--no-mmap`/dio เมื่อ offload ครบ และ mmap เมื่อ RAM ตึงและมี expert อยู่บน CPU (4) estimator สองชั้นสำหรับ MoE (VRAM + RAM) ที่เตือนเมื่อ RAM ไม่พอ ซึ่ง `--fit` ไม่ทำ (5) แสดง "context สูงสุดบน GPU" เป็น output (6) ตรวจ Windows sysmem spill | #21690, PR 16391, man page, discussion 15396, fit README |
| **0.3 Cloud interop + routing** | (1) ส่งต่อ `/v1/messages` และ `/v1/chat/completions` ของ llama-server ตรงๆ (2) shim บางๆ สำหรับ Ollama `/api/*` และ Gemini `generateContent` (3) router แบบ local-first พร้อม fallback ไป cloud key (แนว LiteLLM) (4) router แบบ threshold สไตล์ RouteLLM-MF ที่ calibrate จากคะแนนภาษาที่วัดเอง เช่น ส่งภาษาไทยหรือ math ยากไป cloud เมื่อ quant local ต่ำกว่า floor | HF blog, Ollama docs, LiteLLM, RouteLLM |
| **0.4 GPU speed** | (1) "squeeze step": รัน `llama-bench` pp512/tg128 ทุก backend ที่มี (CUDA/HIP/Vulkan/SYCL) × FA on/off แล้ว cache ผลตาม GPU+driver+shape (2) ตรวจ ReBAR / host-visible heap เล็ก แล้วตั้ง `GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1` อัตโนมัติพร้อมยืนยันด้วย A/B (3) เปิด `GGML_CUDA_GRAPH_OPT=1` บน NVIDIA ที่มี GPU เดียว (4) เปิด MTP อัตโนมัติเมื่อ GGUF มี MTP tensor แล้ว A/B n_max 2/3/5 และเปิดค้างเฉพาะเมื่อเร็วขึ้นเกิน 1.1x (5) เลือก backend ตามงาน: prefill-heavy (RAG) กับ decode-heavy (chat) (6) sweep `-b/-ub` เอง | am17an, #27097, MTP sources, glukhov |
| **Compression research** | (1) ตั้ง floor ที่ UD-Q2_K_XL และเตือนเมื่อต่ำกว่า (2) จัดอันดับ checkpoint QAT ของผู้ขาย (Gemma QAT) เหนือ PTQ (3) ทดลองซ้ำชุด multilingual 2-bit ด้วย IQ2_KT/IQ2_KL (ik_llama) และ EXL3 2.5 bpw (4) imatrix ผสมหลายภาษาแบบมี chat template (EN+TH+HI+AR+code+math) เทียบกับ EN-only และ TH-only ที่ 2-bit โดยวรรณกรรมคาดว่าได้แค่ ~+3 (5) recipe แบบ sensitivity-aware ที่วัด KLD ต่อ tensor แล้วส่งออก `--tensor-type` override และเก็บ embedding/output ไว้สูงสำหรับสคริปต์ non-Latin (6) self-distillation แบบ LoRA ของ 2-bit 27B จาก teacher Q8 (อนุมานว่าทำได้บน 24 GB แต่ยังไม่มีใครวัด) (7) metric หลักใช้ **per-language KLD/top-1 เทียบ BF16** ไม่ใช่ English PPL | Unsloth, PR #113, Kaitchup, 2508.20893, 2601.18306, ParetoQ, 2407.03211 |

## Elysia กับ LLM serving: "อย่าโหลด อย่าก๊อป สิ่งที่ไม่ได้ใช้"

Elysia ใช้ static code analysis (Sucrose) ดูว่าแต่ละ handler ใช้อะไรจริงบ้าง แล้วสร้างโค้ดแบบ JIT ที่ตัดส่วนที่ไม่จำเป็นออก ([at a glance](https://elysiajs.com/at-glance.html)) ใน 1.2 ทีมได้ refactor ให้ใช้ code generation ร่วมกันแทนการสร้างโค้ดซ้ำต่อ route และใช้ checksum hash map ตรวจว่า route ลงทะเบียนแล้วหรือยัง ทำให้ใช้ **หน่วยความจำน้อยลงถึง 2x เทียบกับ 1.1** ([Elysia 1.2 blog](https://elysiajs.com/blog/elysia-12)) ตัวเลขนี้เป็นของผู้พัฒนาเอง และ benchmark ใน at-glance วัดตั้งแต่ปี 2023 บน Bun 0.7.2

แต่ละเทคนิคเทียบกับ LLM serving ได้ดังนี้ (ทั้งหมดเป็นการอนุมานเชิงสถาปัตยกรรม)

| เทคนิคใน Elysia | สิ่งที่เทียบได้ใน LLM serving | ใช้ได้วันนี้แค่ไหน |
|---|---|---|
| วิเคราะห์ว่า handler ใช้อะไร แล้วตัดส่วนที่ไม่ใช้ | ตัด vocabulary ให้เหลือเฉพาะภาษาที่ผู้ใช้ใช้จริง และข้าม MTP, vision หรือ audio tower เมื่อไม่ได้ใช้ (เหมือน Unsloth ที่ตัด MTP ออกจาก quant ต่ำ และ conditional loading ของ Gemma 3n) | vocab trim ของ GGUF ยังไม่มีเครื่องมือ ส่วนการข้าม mmproj/MTP ทำได้แล้ว |
| ใช้โค้ดร่วมกันแทนการก๊อปต่อ route | แชร์ prefix cache แบบ block/radix (vLLM, SGLang) แทนการเก็บ state ทั้ง prompt ต่อ slot แบบ llama.cpp | ใน llama.cpp ยังไม่มี ตอนนี้ทำได้แค่ลด `--cache-ram` และ checkpoint |
| ตรวจซ้ำด้วย checksum ก่อนลงทะเบียน | hash prefix แล้ว dedup (vLLM ใช้ SHA256 ต่อ block) | อยู่ใน engine อื่น |
| JIT สร้างโค้ดที่เหมาะกับ runtime | auto-tune ต่อ GPU: เลือก backend, FA, MTP n_max, ReBAR env และ `-b/-ub` แล้ว cache ผล ซึ่งเทียบได้กับ "compile ครั้งเดียวต่อเครื่อง" | ทำได้ใน launcher (แผน 0.4) |

## Conclusion

ภาพรวมเปลี่ยนจาก "บีบให้เล็กลงอีก" เป็น "**วัดให้ถูกภาษา แล้วเลิกเสีย RAM ไปกับค่า default**" ฝั่ง compression มีเพดานทางกลไกที่ ~2-3 บิต (ParetoQ) ต่ำกว่านั้น calibration แทบไม่ช่วย ทางที่เหลือคือ codec ที่ดีกว่า หรือการเทรนซ่อม ซึ่งทั้งสองทางยังติดปัญหาเรื่องรองรับ AMD หรือต้นทุนการทดลอง ส่วนฝั่ง RAM และความเร็ว ผลตอบแทนมากที่สุดมาจากการ**ตั้งค่าและ auto-tune ต่อเครื่อง** ไม่ใช่อัลกอริทึมใหม่ ตัวอย่างคือ checkpoint/cache-ram, ReBAR env, CUDA graph และ MTP A/B

ข้อมูลที่คุณวัดเองบน RDNA4 ทั้ง ReBAR 1.7x และ MTP +40% เป็นของหายากในวรรณกรรมตอนนี้ การเผยแพร่ข้อมูลชุดนี้เป็นฐานข้อมูลเปิดของ "คุณภาพรายภาษา × quant × ความเร็วต่อ GPU" คือการเติมช่องว่างที่ llmfit, LocalScore และ Intel leaderboard ต่างก็ทำได้คนละครึ่ง
