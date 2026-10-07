import json

from localllm import pool, router
from test_gateway import _gw, _post, _up

Q, G = "qwen3.8-27b-q3", "gemma4-26b-a4b-qat"
ZH = "请用三句话解释为什么天空是蓝色的。"
TH = "ช่วยอธิบายว่าทำไมท้องฟ้าถึงเป็นสีฟ้า"


def test_pick_local_switches_only_for_a_clear_gain():
    assert router.pick_local(ZH, [Q, G], G, 15.9)[0] == Q                 # zh: Qwen +5 pts -> worth a swap
    assert router.pick_local(TH, [Q, G], Q, 15.9)[0] == Q                 # th: gemma within 2 pts -> keep loaded
    assert router.pick_local("Hello there, how are you?", [Q, G], None, 15.9)[0] == G   # cold start: tie -> faster
    assert router.pick_local(ZH, [Q], None, 15.9)[0] == Q                 # single model: nothing to pick


class FakeProc:
    def __init__(self, server):
        self.server = server

    def terminate(self):
        self.server.shutdown()

    def wait(self, _t=None):
        return 0


def test_pool_lazy_loads_and_gateway_reports_the_model(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    launched = []

    def launch(key):
        launched.append(key)
        s, url = _up()
        return FakeProc(s), url

    p = pool.Pool([Q, G], launch, 15.9, first=G)
    g, gurl = _gw(p)
    r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": "Hi, what is 2+2 in words?"}]})
    assert r.headers["X-Localllm-Model"].startswith(G) and launched == [G]          # English: stays on gemma
    r = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": ZH}]})
    assert r.headers["X-Localllm-Model"].startswith(Q) and "swapped" in r.headers["X-Localllm-Model"]
    assert json.load(r)["choices"][0]["message"]["content"] == "echo: " + ZH
    assert launched == [G, Q] and len(p.swaps) == 2
    r = _post(gurl + "/api/chat", {"messages": [{"role": "user", "content": ZH}], "stream": False})
    assert r.headers["X-Localllm-Model"].startswith(Q) and launched == [G, Q]      # Ollama path goes through the pool too
    p.close(); g.shutdown()
