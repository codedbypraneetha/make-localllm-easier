import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from localllm import chat

CHUNKS = [
    {"choices": [{"delta": {"reasoning_content": "hmm"}}]},
    {"choices": [{"delta": {"content": "สวัสดี"}}]},
    {"choices": [{"delta": {"content": " ครับ"}}]},
    {"choices": [], "timings": {"predicted_n": 2, "predicted_per_second": 42.0}},
]


class FakeServer(BaseHTTPRequestHandler):
    seen = []

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b'{"status":"ok"}')

    def do_POST(self):
        FakeServer.seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        for c in CHUNKS:
            self.wfile.write(f"data: {json.dumps(c, ensure_ascii=False)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def _serve():
    srv = HTTPServer(("127.0.0.1", 0), FakeServer)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_port}"


def test_stream_prints_answer_hides_thinking_and_returns_timings():
    srv, url = _serve()
    out = io.StringIO()
    answer, t = chat.stream(url, [{"role": "user", "content": "hi"}], show_thinking=False, out=out)
    srv.shutdown()
    assert answer == "สวัสดี ครับ"
    assert "hmm" not in out.getvalue() and "(thinking...)" in out.getvalue()
    assert t["predicted_per_second"] == 42.0


def test_stream_can_show_thinking():
    srv, url = _serve()
    out = io.StringIO()
    chat.stream(url, [{"role": "user", "content": "hi"}], show_thinking=True, out=out)
    srv.shutdown()
    assert "hmm" in out.getvalue()


def test_repl_keeps_history_and_commands(monkeypatch, tmp_path, capsys):
    srv, url = _serve()
    FakeServer.seen.clear()
    save = tmp_path / "c.md"
    inputs = iter(["hello", "again", f"/save {save}", "/clear", "after clear", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
    assert chat.server_alive(url)
    chat.repl(url)
    srv.shutdown()
    assert [len(b["messages"]) for b in FakeServer.seen] == [1, 3, 1]   # history grows, /clear resets
    assert "สวัสดี ครับ" in save.read_text(encoding="utf-8")
    assert "42 tok/s" in capsys.readouterr().out


def test_save_to_bad_path_reports_instead_of_crashing(monkeypatch, capsys):
    srv, url = _serve()
    inputs = iter(["hello", "/save Z:/no/such/dir/x.md", "/exit"])
    monkeypatch.setattr("builtins.input", lambda _p="": next(inputs))
    chat.repl(url)
    srv.shutdown()
    assert "could not save" in capsys.readouterr().out
