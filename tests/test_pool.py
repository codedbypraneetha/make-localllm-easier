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


def test_detect_task():
    assert router.detect_task("Janet has 16 eggs, eats 3 and bakes with 4. How many are left?") == "math"
    assert router.detect_task("เป็ดวางไข่วันละ 16 ฟอง กินไป 3 ฟอง เหลือกี่ฟอง") == "math"
    assert router.detect_task("What is 17 * 23?") == "math"
    assert router.detect_task("Write a poem about the sea.") == "general"
    assert router.detect_task("I was born in 1990 and moved in 2010.") == "general"   # numbers alone aren't math


def test_task_changes_the_pick_for_chinese():
    zh_math = "小明有 15 个苹果，给了朋友 7 个，又买了 12 个。他现在有多少个苹果？"
    assert router.pick_local(ZH, [Q, G], G, 15.9)[0] == Q          # zh knowledge: Qwen +5.5
    assert router.pick_local(zh_math, [Q, G], Q, 15.9)[0] == G     # zh math: gemma +4.4 (MGSM)


def test_resident_pool_routes_without_swaps(monkeypatch):
    monkeypatch.setattr(router, "load_config", lambda: {"enabled": False})
    launched = []

    def launch(key):
        launched.append(key)
        s, url = _up()
        return FakeProc(s), url

    p = pool.Pool([Q, G], launch, 32.0, first=G, resident=True)
    g, gurl = _gw(p)
    assert sorted(launched) == sorted([Q, G]) and p.swaps == []
    th = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": TH}]})
    zh = _post(gurl + "/v1/chat/completions", {"messages": [{"role": "user", "content": ZH}]})
    assert th.headers["X-Localllm-Model"].startswith(G) and zh.headers["X-Localllm-Model"].startswith(Q)
    assert "swapped" not in zh.headers["X-Localllm-Model"] and len(launched) == 2
    p.close(); g.shutdown()


def test_detect_translate():
    assert router.detect_task("Translate into Thai: The meeting is at 3 pm.") == "translate"
    assert router.detect_task("ช่วยแปลประโยคนี้เป็นภาษาอังกฤษ: วันนี้อากาศดี") == "translate"
    assert router.detect_task("请把这句话翻译成英文：今天天气很好。") == "translate"
    assert router.detect_task("How many people speak Thai?") == "general"
