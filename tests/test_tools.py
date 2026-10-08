"""Tool / function calling through all four APIs (#45), against a fake llama-server that speaks OpenAI tools."""
import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from localllm import gateway, router

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
                                            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                                           "required": ["city"]}}}


class FakeToolLlama(BaseHTTPRequestHandler):
    """Asks for get_weather while the conversation has tools and no tool result yet; otherwise answers in text."""
    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def _send(self, obj, stream):
        if not stream:
            out = json.dumps(obj).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
            return
        self.send_response(200); self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
        for c in obj:
            d = f"data: {json.dumps(c)}\n\n".encode()
            self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n")
        d = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n0\r\n\r\n")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeToolLlama.seen.append((self.path, body))
        msgs = body.get("messages", [])
        call = body.get("tools") and not any(m.get("role") == "tool" for m in msgs)
        stream = body.get("stream")
        if call:
            if stream:           # arguments arrive in fragments, the way llama-server streams them
                return self._send([
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "abc", "type": "function",
                                                            "function": {"name": "get_weather", "arguments": ""}}]}}]},
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '{"city": '}}]}}]},
                    {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": '"Bangkok"}'}}]}}]},
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}], True)
            return self._send({"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "abc", "type": "function",
                 "function": {"name": "get_weather", "arguments": '{"city": "Bangkok"}'}}]},
                "finish_reason": "tool_calls"}]}, False)
        result = next((m["content"] for m in reversed(msgs) if m.get("role") == "tool"), "")
        text = f"It is {result}"
        if stream:
            return self._send([{"choices": [{"delta": {"content": text}}]},
                               {"choices": [{"delta": {}, "finish_reason": "stop"}]}], True)
        return self._send({"choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}]},
                          False)


def _servers(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    up = HTTPServer(("127.0.0.1", 0), FakeToolLlama)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    g = gateway.serve(f"http://127.0.0.1:{up.server_port}", port=0, model_name="m")
    return up, g, f"http://127.0.0.1:{g.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def test_openai_and_anthropic_tools_pass_through_unchanged(monkeypatch):
    up, g, url = _servers(monkeypatch)
    try:
        r = json.loads(_post(url + "/v1/chat/completions", {"messages": [{"role": "user", "content": "weather?"}],
                                                             "tools": [WEATHER]}))
        assert FakeToolLlama.seen[-1][1]["tools"] == [WEATHER]
        assert r["choices"][0]["message"]["tool_calls"][0]["function"]["name"] == "get_weather"
        anth = {"model": "m", "max_tokens": 50, "messages": [{"role": "user", "content": "weather?"}],
                "tools": [{"name": "get_weather", "input_schema": WEATHER["function"]["parameters"]}]}
        _post(url + "/v1/messages", anth)
        assert FakeToolLlama.seen[-1] == ("/v1/messages", anth)          # llama-server translates Anthropic itself
    finally:
        g.shutdown(); up.shutdown()


def test_ollama_tool_round_trip_non_stream_and_stream(monkeypatch):
    up, g, url = _servers(monkeypatch)
    try:
        for stream in (False, True):
            ask = {"model": "m", "stream": stream, "tools": [WEATHER],
                   "messages": [{"role": "user", "content": "weather in Bangkok?"}]}
            lines = [json.loads(x) for x in _post(url + "/api/chat", ask).splitlines() if x.strip()]
            assert FakeToolLlama.seen[-1][1]["tools"] == [WEATHER]
            last = lines[-1]
            assert last["done"] and last["message"]["tool_calls"] == \
                [{"function": {"name": "get_weather", "arguments": {"city": "Bangkok"}}}]   # object, not a string
            answer = {"model": "m", "stream": stream, "tools": [WEATHER], "messages": [
                *ask["messages"], last["message"], {"role": "tool", "content": "31°C", "tool_name": "get_weather"}]}
            lines = [json.loads(x) for x in _post(url + "/api/chat", answer).splitlines() if x.strip()]
            sent = FakeToolLlama.seen[-1][1]["messages"]
            assert sent[1]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Bangkok"}'}
            assert sent[2] == {"role": "tool", "content": "31°C", "tool_call_id": sent[1]["tool_calls"][0]["id"]}
            assert "".join(x["message"]["content"] for x in lines) == "It is 31°C"
    finally:
        g.shutdown(); up.shutdown()


def test_gemini_function_calling_round_trip_non_stream_and_stream(monkeypatch):
    up, g, url = _servers(monkeypatch)
    decl = {"functionDeclarations": [{"name": "get_weather", "description": "Weather for a city",
                                      "parameters": {"type": "OBJECT", "properties": {"city": {"type": "STRING"}}}}]}
    try:
        for method in ("generateContent", "streamGenerateContent"):
            ask = {"contents": [{"role": "user", "parts": [{"text": "weather in Bangkok?"}]}], "tools": [decl],
                   "toolConfig": {"functionCallingConfig": {"mode": "ANY"}}}
            raw = _post(url + f"/v1beta/models/m:{method}", ask)
            chunks = [json.loads(x[5:]) for x in raw.splitlines() if x.startswith("data:")] or [json.loads(raw)]
            sent = FakeToolLlama.seen[-1][1]
            assert sent["tools"][0]["function"]["parameters"] == \
                {"type": "object", "properties": {"city": {"type": "string"}}}         # Gemini's upper-case types
            assert sent["tool_choice"] == "required"
            parts = [p for c in chunks for p in c["candidates"][0]["content"]["parts"]]
            assert {"functionCall": {"name": "get_weather", "args": {"city": "Bangkok"}}} in parts
            answer = {"contents": [*ask["contents"], {"role": "model", "parts": [
                {"functionCall": {"name": "get_weather", "args": {"city": "Bangkok"}}}]},
                {"role": "user", "parts": [{"functionResponse": {"name": "get_weather",
                                                                 "response": {"temp": "31°C"}}}]}], "tools": [decl]}
            raw = _post(url + f"/v1beta/models/m:{method}", answer)
            chunks = [json.loads(x[5:]) for x in raw.splitlines() if x.startswith("data:")] or [json.loads(raw)]
            msgs = FakeToolLlama.seen[-1][1]["messages"]
            assert msgs[1]["tool_calls"][0]["function"] == {"name": "get_weather", "arguments": '{"city": "Bangkok"}'}
            assert msgs[2]["role"] == "tool" and msgs[2]["tool_call_id"] == msgs[1]["tool_calls"][0]["id"]
            assert json.loads(msgs[2]["content"]) == {"temp": "31°C"}
            text = "".join(p.get("text", "") for c in chunks for p in c["candidates"][0]["content"]["parts"])
            assert text == 'It is {"temp": "31°C"}'
    finally:
        g.shutdown(); up.shutdown()


def test_tool_call_deltas_are_merged_by_index():
    t = gateway.ToolCalls()
    for d in ({"tool_calls": [{"index": 1, "id": "b", "function": {"name": "f2", "arguments": "{}"}}]},
              {"tool_calls": [{"index": 0, "id": "a", "function": {"name": "f1", "arguments": '{"x"'}}]},
              {"tool_calls": [{"index": 0, "function": {"arguments": ": 1}"}}]}, {"content": "hi"}):
        t.add(d)
    assert [(c["id"], c["function"]["name"], c["function"]["arguments"]) for c in t.calls()] == \
        [("a", "f1", '{"x": 1}'), ("b", "f2", "{}")]
    assert gateway._args_obj('{"x": 1}') == {"x": 1} and gateway._args_obj("not json") == {"_raw": "not json"}
