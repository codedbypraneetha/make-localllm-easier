"""Does a model call tools correctly through each of localllm's four APIs? (#45)

  python tools/tool_check.py --llama http://127.0.0.1:8081 [--name MODEL] [--json out.jsonl]

Starts the localllm gateway in front of a running llama-server and, for OpenAI, Anthropic, Ollama and Gemini, streaming
and not: asks for the weather with a get_weather tool, checks the model calls it with a city, sends the tool result
back and checks the answer uses it. Prints one row per API and mode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localllm import gateway  # noqa: E402

ASK = "What's the weather in Bangkok right now? Use the tool."
RESULT = {"city": "Bangkok", "temp_c": 31, "sky": "sunny"}
PARAMS = {"type": "object", "properties": {"city": {"type": "string", "description": "city name"}}, "required": ["city"]}
OPENAI_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city",
                                                "parameters": PARAMS}}


def post(url: str, body: dict, headers: dict | None = None) -> str:
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=600) as r:
        return r.read().decode()


def sse(raw: str) -> list[dict]:
    return [json.loads(x[5:]) for x in raw.splitlines() if x.startswith("data:") and x[5:].strip() != "[DONE]"]


def openai(url: str, stream: bool) -> tuple[dict | None, str]:
    msgs = [{"role": "user", "content": ASK}]
    body = {"messages": msgs, "tools": [OPENAI_TOOL], "max_tokens": 512, "stream": stream}
    raw = post(url + "/v1/chat/completions", body)
    if stream:
        t = gateway.ToolCalls()
        for c in sse(raw):
            t.add((c.get("choices") or [{}])[0].get("delta", {}))
        calls = t.calls()
    else:
        calls = json.loads(raw)["choices"][0]["message"].get("tool_calls") or []
    if not calls:
        return None, ""
    c = calls[0]
    msgs += [{"role": "assistant", "content": "", "tool_calls": [c]},
             {"role": "tool", "tool_call_id": c["id"], "content": json.dumps(RESULT)}]
    raw = post(url + "/v1/chat/completions", {**body, "messages": msgs})
    text = "".join((c.get("choices") or [{}])[0].get("delta", {}).get("content") or "" for c in sse(raw)) if stream \
        else json.loads(raw)["choices"][0]["message"].get("content") or ""
    return {"name": c["function"]["name"], "args": gateway._args_obj(c["function"]["arguments"])}, text


def anthropic(url: str, stream: bool) -> tuple[dict | None, str]:
    tool = {"name": "get_weather", "description": "Current weather for a city", "input_schema": PARAMS}
    msgs = [{"role": "user", "content": ASK}]
    body = {"model": "local", "max_tokens": 512, "messages": msgs, "tools": [tool], "stream": stream}
    raw = post(url + "/v1/messages", body, {"anthropic-version": "2023-06-01"})
    if stream:
        use, args = None, ""
        for e in sse(raw):
            if e.get("type") == "content_block_start" and e["content_block"].get("type") == "tool_use":
                use = dict(e["content_block"])
            elif e.get("type") == "content_block_delta" and e["delta"].get("type") == "input_json_delta":
                args += e["delta"].get("partial_json", "")
        if use:
            use["input"] = gateway._args_obj(args)
    else:
        use = next((b for b in json.loads(raw)["content"] if b.get("type") == "tool_use"), None)
    if not use:
        return None, ""
    msgs += [{"role": "assistant", "content": [use]},
             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use["id"],
                                           "content": json.dumps(RESULT)}]}]
    raw = post(url + "/v1/messages", {**body, "messages": msgs}, {"anthropic-version": "2023-06-01"})
    text = "".join(e["delta"].get("text", "") for e in sse(raw) if e.get("type") == "content_block_delta") if stream \
        else "".join(b.get("text", "") for b in json.loads(raw)["content"] if b.get("type") == "text")
    return {"name": use["name"], "args": use.get("input") or {}}, text


def ollama(url: str, stream: bool) -> tuple[dict | None, str]:
    msgs = [{"role": "user", "content": ASK}]
    body = {"model": "local", "stream": stream, "tools": [OPENAI_TOOL], "messages": msgs,
            "options": {"num_predict": 512}}
    lines = [json.loads(x) for x in post(url + "/api/chat", body).splitlines() if x.strip()]
    calls = [c for x in lines for c in x.get("message", {}).get("tool_calls") or []]
    if not calls:
        return None, ""
    msgs += [{"role": "assistant", "content": "", "tool_calls": calls[:1]},
             {"role": "tool", "content": json.dumps(RESULT), "tool_name": "get_weather"}]
    lines = [json.loads(x) for x in post(url + "/api/chat", {**body, "messages": msgs}).splitlines() if x.strip()]
    f = calls[0]["function"]
    return {"name": f["name"], "args": f.get("arguments") or {}}, \
        "".join(x.get("message", {}).get("content", "") for x in lines)


def gemini(url: str, stream: bool) -> tuple[dict | None, str]:
    decl = {"functionDeclarations": [{"name": "get_weather", "description": "Current weather for a city",
                                      "parameters": {"type": "OBJECT", "properties": {"city": {"type": "STRING"}},
                                                     "required": ["city"]}}]}
    contents = [{"role": "user", "parts": [{"text": ASK}]}]
    method = "streamGenerateContent?alt=sse" if stream else "generateContent"
    body = {"contents": contents, "tools": [decl], "generationConfig": {"maxOutputTokens": 512}}

    def parts(raw: str) -> list[dict]:
        chunks = sse(raw) if stream else [json.loads(raw)]
        return [p for c in chunks for p in c["candidates"][0]["content"].get("parts", [])]

    call = next((p["functionCall"] for p in parts(post(url + f"/v1beta/models/local:{method}", body))
                 if "functionCall" in p), None)
    if not call:
        return None, ""
    contents += [{"role": "model", "parts": [{"functionCall": call}]},
                 {"role": "user", "parts": [{"functionResponse": {"name": call["name"], "response": RESULT}}]}]
    raw = post(url + f"/v1beta/models/local:{method}", {**body, "contents": contents})
    return {"name": call["name"], "args": call.get("args") or {}}, "".join(p.get("text", "") for p in parts(raw))


def check(url: str) -> list[dict]:
    rows = []
    for api, fn in (("openai", openai), ("anthropic", anthropic), ("ollama", ollama), ("gemini", gemini)):
        for stream in (False, True):
            t = time.time()
            try:
                call, text = fn(url, stream)
                err = ""
            except Exception as e:  # noqa: BLE001 - one broken API must not hide the others
                call, text, err = None, "", f"{type(e).__name__}: {e}"[:200]
            city = str((call or {}).get("args", {}).get("city", ""))
            rows.append({"api": api, "stream": stream, "called": bool(call),
                         "valid": bool(call) and call.get("name") == "get_weather" and "bangkok" in city.lower(),
                         "used_result": "31" in text, "seconds": round(time.time() - t, 1), "error": err,
                         "call": call, "answer": text[:160]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--llama", required=True, help="URL of a running llama-server")
    ap.add_argument("--name", default="model")
    ap.add_argument("--json", type=Path, help="append the rows here as JSON lines")
    a = ap.parse_args()
    g = gateway.serve(a.llama, port=0, model_name=a.name)
    rows = check(f"http://127.0.0.1:{g.server_address[1]}")
    g.shutdown()
    for r in rows:
        mark = "OK  " if r["valid"] and r["used_result"] else ("CALL" if r["valid"] else "FAIL")
        print(f"{mark} {a.name:<20} {r['api']:<9} {'stream' if r['stream'] else 'json  '} "
              f"call={json.dumps(r['call'], ensure_ascii=False)} answer={r['answer']!r} {r['error']}")
        if a.json:
            with open(a.json, "a", encoding="utf-8") as f:
                f.write(json.dumps({"model": a.name, **r}, ensure_ascii=False) + "\n")
    if all(r["error"] for r in rows):
        sys.exit("every API errored: is llama-server running?")


if __name__ == "__main__":
    main()
