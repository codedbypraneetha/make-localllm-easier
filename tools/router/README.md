# Embedding task router (issue #28) - research scripts

Goal: classify each message as `general` / `math` / `code` / `translate` in any language, in < 20 ms, with a model far
smaller than Laya (614 MB), then let the measured catalog scores pick the model.

| file | what |
|---|---|
| `check_embed.py` | llama.cpp GGUF embeddings vs PyTorch (cosine 0.992-0.9998 for e5-small Q8_0) |
| `build_data.py` | training data labelled by source (MGSM, GSM8K, MBPP, CRUXEval, FLORES, Aya, oasst2, Global-MMLU, ThaiExam) |
| `gen_data.py` | short everyday-style requests per class in 15 languages, written by a local LLM |
| `train_router.py` | embed with llama-server `--embedding --pooling mean`, logistic-regression head, eval vs keyword rules |
| `eval_laya.py` | Laya (multilingual checkpoint, zero-shot choice) on the same test set |
| `shared_test.jsonl` | 144 hand-written requests, 15 languages, never used for training |

Model: `intfloat/multilingual-e5-small` (MIT) converted with `convert_hf_to_gguf.py` after setting
`architectures: ["XLMRobertaModel"]` and removing `pad_token_id` (BERT body + XLM-R sentencepiece tokenizer; keeping
pad_token_id would shift position embeddings by one). Q8_0 = 126 MB. Inputs need the `query: ` prefix, cap ~450 chars.

**Current result** (training data + 2,833 short everyday requests in 15 languages written by gemma-4 with `gen_data.py`;
`train_router.py --with-generated`; head in `router_head.json`, 4 x 384 logistic regression on e5-small Q8_0):

| | shared test (144) | items least like training (nearest cosine < 0.90, n=73) | latency, CPU | size |
|---|---|---|---|---|
| **embedding router** | **97.9%** | **98.6%** | 7.3 ms median | 126 MB |
| Laya multilingual (zero-shot) | 77.8% | 71.2% | 103 ms median | 614 MB |
| keyword rules (`detect_task`) | 56.9% | - | < 1 ms | 0 |

Caveat: the shared test set is written by an AI (Claude) too, and generated training items are AI-written; 10 of 144 test
items have a training item with cosine > 0.95 (the filtered column removes such cases). A human-written test set from
native speakers is the next proof needed.

First results, before the generated data (RX 9070 XT PC, CPU, shared test): Laya 77.8% (103 ms median) | embedding head 64.6% | embedding head with
keyword rules first 72.2% (~6 ms) | keyword rules alone 56.9%. Held-out (same style as training) 98.6%: the gap is
training-data style, being fixed with generated short requests.
