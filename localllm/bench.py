"""Multilingual multiple-choice benchmark for any OpenAI-compatible server (llama-server, Ollama, LM Studio, vLLM ...).

Two kinds of test per language, so a score means something wherever you live:
  global    Global-MMLU-Lite (CohereLabs, Apache-2.0): the same 400 questions translated into 23 languages,
            so languages and models compare like for like
  regional  INCLUDE-lite-44 (CohereLabs, Apache-2.0): real exams written in each country (~250 per language,
            44 languages); ThaiExam (typhoon-ai, Apache-2.0) fills in Thai
Zero-shot, thinking off, one token: the answer is the option letter with the highest log-probability. Fast
(prompt processing only) and deterministic. Data is downloaded at eval time and cached, never redistributed.

Task suites (opt-in with --suites, they generate text so they are slower):
  math      MGSM (Shi et al. 2022, CC-BY-SA-4.0): the same 250 grade-school word problems in 11 languages; the model
            reasons in text (thinking off) and the final number is compared exactly
"""
from __future__ import annotations

import ast
import json
import locale
import time
import urllib.parse
import urllib.request

from .runtime import HOME

ROWS = "https://datasets-server.huggingface.co/rows?dataset={ds}&config={cfg}&split={split}&offset={off}&length=100"
GLOBAL_LANGS = ["ar", "bn", "cs", "cy", "de", "en", "es", "fr", "hi", "hu", "id", "it", "ja", "ko", "my", "or", "pt",
                "sk", "sq", "sw", "tg", "yo", "zh"]
INCLUDE = {"sq": "Albanian", "ar": "Arabic", "hy": "Armenian", "az": "Azerbaijani", "eu": "Basque", "be": "Belarusian",
           "bn": "Bengali", "bg": "Bulgarian", "zh": "Chinese", "hr": "Croatian", "nl": "Dutch", "et": "Estonian",
           "fi": "Finnish", "fr": "French", "ka": "Georgian", "de": "German", "el": "Greek", "he": "Hebrew",
           "hi": "Hindi", "hu": "Hungarian", "id": "Indonesian", "it": "Italian", "ja": "Japanese", "kk": "Kazakh",
           "ko": "Korean", "lt": "Lithuanian", "ms": "Malay", "ml": "Malayalam", "ne": "Nepali",
           "mk": "North Macedonian", "fa": "Persian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian",
           "sr": "Serbian", "es": "Spanish", "tl": "Tagalog", "ta": "Tamil", "te": "Telugu", "tr": "Turkish",
           "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese"}
THAIEXAM = "https://huggingface.co/datasets/typhoon-ai/thai_exam/resolve/main/data/{s}/{s}_test.jsonl"
SYSTEM = "Answer the multiple-choice question. Reply with only the letter of the correct option."
MGSM_LANGS = ["bn", "de", "en", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"]
MATH_SYSTEM = "Solve the problem step by step, briefly. End with a last line of the form 'Answer: <number>'."
SUITES = ("global", "regional", "math")


def system_language() -> str:
    try:
        loc = locale.getlocale()[0] or ""
    except ValueError:
        loc = ""
    if "_" in loc or len(loc) == 2:
        return loc.split("_")[0].lower()[:2]
    names = {n.lower(): c for c, n in INCLUDE.items()} | {"thai": "th", "english": "en"}
    return names.get(loc.split("_")[0].lower(), "en")


def available(lang: str, suites: tuple[str, ...] = ("global", "regional")) -> list[str]:
    have = {"global": lang in GLOBAL_LANGS, "regional": lang in INCLUDE or lang == "th", "math": lang in MGSM_LANGS}
    return [s for s in suites if have[s]]


def _rows(ds: str, cfg: str, split: str = "test") -> list[dict]:
    out, off = [], 0
    while True:
        u = ROWS.format(ds=urllib.parse.quote(ds), cfg=urllib.parse.quote(cfg), split=split, off=off)
        d = json.load(urllib.request.urlopen(u, timeout=120))
        out += [r["row"] for r in d["rows"]]
        off += 100
        if off >= d.get("num_rows_total", 0) or not d["rows"]:
            return out


def load(suite: str, lang: str) -> list[dict]:
    """[{'q': question, 'opts': [..], 'ans': index}] cached under ~/.localllm/bench/."""
    cache = HOME / "bench" / f"{suite}-{lang}.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    items = []
    if suite == "global":
        for r in _rows("CohereLabs/Global-MMLU-Lite", lang):
            opts = [r[f"option_{c}"] for c in "abcd"]
            items.append({"q": r["question"], "opts": opts, "ans": "ABCD".index(r["answer"].strip().upper())})
    elif suite == "regional" and lang == "th":
        for s in ["onet", "ic", "tgat", "tpat1", "a_level"]:
            for line in urllib.request.urlopen(THAIEXAM.format(s=s)).read().decode("utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    keys = [c for c in "abcde" if r.get(c)]
                    items.append({"q": r["question"], "opts": [r[c] for c in keys],
                                  "ans": keys.index(r["answer"].strip().lower())})
    elif suite == "math" and lang in MGSM_LANGS:
        for r in _rows("juletxara/mgsm", lang):
            items.append({"q": r["question"], "ans": int(r["answer_number"])})
    elif suite == "regional":
        for r in _rows("CohereLabs/include-lite-44", INCLUDE[lang]):
            opts = r["choices"] if isinstance(r["choices"], list) else ast.literal_eval(r["choices"])
            items.append({"q": r["question"], "opts": opts, "ans": int(r["answer"])})
    else:
        raise ValueError(f"no {suite} test for '{lang}'")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return items


def ask(url: str, item: dict) -> int | None:
    letters = "ABCDEFGHIJ"[: len(item["opts"])]
    opts = "\n".join(f"{l}. {o}" for l, o in zip(letters, item["opts"]))
    body = {"messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": f"{item['q']}\n\n{opts}\n\nAnswer ({'/'.join(letters)}):"}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(req, timeout=600))
    best, best_lp = None, -1e9
    for t in d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        tok = t["token"].strip().upper().strip(".()")
        if len(tok) == 1 and tok in letters and t["logprob"] > best_lp:
            best, best_lp = letters.index(tok), t["logprob"]
    return best


def final_number(text: str) -> float | None:
    """The answer of a worked solution: the number after the last 'Answer:', else the last number in the text."""
    import re
    tail = text.rsplit("Answer", 1)[-1] if "Answer" in text else text
    nums = re.findall(r"-?\d[\d,]*\.?\d*", tail.replace(" ", " ").replace("**", ""))
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", "").rstrip("."))
    except ValueError:
        return None


def ask_math(url: str, item: dict) -> bool:
    body = {"messages": [{"role": "system", "content": MATH_SYSTEM}, {"role": "user", "content": item["q"]}],
            "max_tokens": 600, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    text = json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["message"].get("content") or ""
    n = final_number(text)
    return n is not None and abs(n - item["ans"]) < 1e-6


def _save(name: str, res: dict) -> None:
    out = HOME / "results.json"
    allres = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    allres[name] = {**allres.get(name, {}), **res}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(allres, ensure_ascii=False, indent=1), encoding="utf-8")


def run(url: str, name: str, langs: list[str], limit: int = 0, suites: tuple[str, ...] = ("global", "regional")) -> dict:
    """Scores are saved after every test, so an interrupted run keeps what it finished."""
    res, t0 = {}, time.time()
    for lang in langs:
        mine = available(lang, suites)
        if not mine:
            print(f"  {lang}: no {'/'.join(suites)} benchmark yet (contributions welcome)")
        for suite in mine:
            items = load(suite, lang)
            items = items[:limit] if limit else items
            ok = sum(ask_math(url, it) if suite == "math" else ask(url, it) == it["ans"] for it in items)
            acc = round(100 * ok / len(items), 1)
            res[f"{lang}/{suite}"] = {"acc": acc, "correct": ok, "n": len(items)}
            margin = round(196 * (acc / 100 * (1 - acc / 100) / len(items)) ** 0.5, 1)
            print(f"  {lang:3} {suite:9} {acc:5.1f}%  ±{margin}  ({ok}/{len(items)})", flush=True)
            _save(name, {**res, "_meta": {"model": name, "seconds": round(time.time() - t0), "limit": limit}})
    print(f"saved to {HOME / 'results.json'}  (share it: open a PR adding your GPU's numbers)")
    return res
