import json
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

from localllm import gateway, router


class FakeLlama(BaseHTTPRequestHandler):
    """Stands in for llama-server (and for a cloud provider): echoes the last user message."""
    protocol_version = "HTTP/1.1"
    seen = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        b = b'{"status":"ok"}'
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLlama.seen.append((self.path, dict(self.headers), body))
        last = body.get("messages", [{}])[-1].get("content", "")
        if body.get("stream"):
            self.send_response(200); self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked"); self.end_headers()
            for c in [{"choices": [{"delta": {"content": "echo: "}}]}, {"choices": [{"delta": {"content": last}}]},
                      {"choices": [{"delta": {}, "finish_reason": "stop"}], "timings": {"predicted_n": 2, "predicted_ms": 10}}]:
                d = f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode()
                self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n")
            d = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(d):x}\r\n".encode() + d + b"\r\n0\r\n\r\n")
            return
        out = json.dumps({"choices": [{"message": {"role": "assistant", "content": "echo: " + last},
                                       "finish_reason": "stop"}],
                          "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


def _up():
    s = HTTPServer(("127.0.0.1", 0), FakeLlama)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s, f"http://127.0.0.1:{s.server_port}"


def _gw(upstream):
    g = gateway.serve(upstream, port=0, model_name="test-model")
    return g, f"http://127.0.0.1:{g.server_address[1]}"


def _post(url, body):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=30)


def test_openai_passthrough_reports_local_route(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    up, uurl = _up(); g, gurl = _gw(uurl)
    r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    assert json.load(r)["choices"][0]["message"]["content"] == "echo: hi" and r.headers["X-Localllm-Route"] == "local"
    g.shutdown(); up.shutdown()


def test_ollama_chat_non_stream_and_stream(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    up, uurl = _up(); g, gurl = _gw(uurl)
    d = json.load(_post(gurl + "/api/chat", {"model": "x", "stream": False,
                                             "messages": [{"role": "user", "content": "สวัสดี"}]}))
    assert d["message"]["content"] == "echo: สวัสดี" and d["done"] is True and d["eval_count"] == 2
    lines = [json.loads(l) for l in _post(gurl + "/api/generate", {"prompt": "yo"}).read().decode().splitlines() if l]
    assert "".join(l.get("response", "") for l in lines) == "echo: yo" and lines[-1]["done"] is True
    tags = json.load(urllib.request.urlopen(gurl + "/api/tags"))
    assert tags["models"][0]["name"] == "test-model"
    g.shutdown(); up.shutdown()


def test_gemini_generate_and_stream(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    up, uurl = _up(); g, gurl = _gw(uurl)
    body = {"contents": [{"role": "user", "parts": [{"text": "hello"}]}], "generationConfig": {"maxOutputTokens": 9}}
    d = json.load(_post(gurl + "/v1beta/models/m:generateContent", body))
    assert d["candidates"][0]["content"]["parts"][0]["text"] == "echo: hello"
    assert d["usageMetadata"]["totalTokenCount"] == 5 and FakeLlama.seen[-1][2]["max_tokens"] == 9
    raw = _post(gurl + "/v1beta/models/m:streamGenerateContent?alt=sse", body).read().decode()
    texts = [json.loads(l[5:])["candidates"][0]["content"]["parts"][0]["text"] for l in raw.splitlines() if l.startswith("data:")]
    assert "".join(texts) == "echo: hello"
    g.shutdown(); up.shutdown()


def test_router_local_by_default_and_rules(monkeypatch):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    assert router.decide(body, "/v1/chat/completions", {"enabled": False}).provider is None
    monkeypatch.setenv("FAKE_KEY", "k")
    cfg = {"enabled": True, "provider": "c",
           "providers": {"c": {"kind": "openai", "url": "http://x", "key_env": "FAKE_KEY", "model": "m"}},
           "rules": {"max_local_prompt_tokens": 50, "language_floor": 70}, "local_model": "qwen3.8-27b-q3"}
    assert router.decide(body, "/v1/chat/completions", cfg).provider is None                       # stays local
    assert router.decide({"model": "gpt-5", **body}, "/v1/chat/completions", cfg).provider.name == "c"
    assert router.decide({"messages": [{"role": "user", "content": "x" * 400}]}, "/v1/chat/completions", cfg).provider
    thai = {"messages": [{"role": "user", "content": "ช่วยอธิบายเรื่องนี้หน่อย"}]}              # th score 67.1 < 70
    assert "th score" in router.decide(thai, "/v1/chat/completions", cfg).label
    assert router.decide(thai, "/v1/messages", cfg).provider is None                               # no anthropic key


def test_router_forwards_to_cloud_with_key_and_model(monkeypatch):
    up, uurl = _up(); cloud, curl = _up(); g, gurl = _gw(uurl)
    monkeypatch.setenv("FAKE_KEY", "secret")
    cfg = {"enabled": True, "providers": {"c": {"kind": "openai", "url": curl, "key_env": "FAKE_KEY", "model": "big"}}}
    monkeypatch.setattr(router, "load_config", lambda: cfg)
    r = _post(gurl + "/v1/chat/completions", {"model": "gpt-5", "messages": [{"role": "user", "content": "q"}]})
    path, headers, body = FakeLlama.seen[-1]
    assert r.headers["X-Localllm-Route"].startswith("cloud:c") and headers["Authorization"] == "Bearer secret"
    assert body["model"] == "big"
    g.shutdown(); up.shutdown(); cloud.shutdown()


def test_detect_language():
    assert router.detect_language("สวัสดีครับ") == "th" and router.detect_language("こんにちは") == "ja"
    assert router.detect_language("hello world") == "en"
