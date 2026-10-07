"""Measure llama-server's system RAM and CPU over a long multi-turn chat (Windows; issues #1 and #33).

  python tools/measure_footprint.py --model PATH.gguf --label default --turns 30 [-- extra llama-server args]

Starts llama-server with localllm's launch args (or llama.cpp defaults with --raw), runs a growing conversation, and
after every turn records the server's private bytes, working set and CPU seconds. Also samples CPU while idle.
Writes one JSON line per run to tools/footprint_results.jsonl.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import time
import urllib.request
from ctypes import wintypes
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from localllm import runtime  # noqa: E402


class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD), ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t), ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t), ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t), ("PrivateUsage", ctypes.c_size_t)]


def proc_stats(pid: int) -> dict:
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x0410, False, pid)          # QUERY_INFORMATION | VM_READ
    try:
        pmc = PMC(); pmc.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), pmc.cb)
        c, e, k, u = (wintypes.FILETIME() for _ in range(4))
        k32.GetProcessTimes(h, ctypes.byref(c), ctypes.byref(e), ctypes.byref(k), ctypes.byref(u))
        ft = lambda f: (f.dwHighDateTime << 32 | f.dwLowDateTime) / 1e7
        return {"private_gb": pmc.PrivateUsage / 2**30, "working_set_gb": pmc.WorkingSetSize / 2**30,
                "peak_ws_gb": pmc.PeakWorkingSetSize / 2**30, "cpu_s": ft(k) + ft(u)}
    finally:
        k32.CloseHandle(h)


TOPICS = ["the history of the printing press", "how vaccines train the immune system", "why the sky is blue",
          "ประวัติศาสตร์ของกรุงศรีอยุธยา", "the economics of renewable energy", "how compilers optimise loops",
          "ระบบสุริยะและดาวเคราะห์แต่ละดวง", "the causes of the 2008 financial crisis", "how GPS works",
          "the life cycle of stars"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--label", required=True)
    ap.add_argument("--turns", type=int, default=30); ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--server", default=str(runtime.find_server()))
    ap.add_argument("--raw", action="store_true", help="llama.cpp defaults for caches (no localllm profile)")
    ap.add_argument("--no-vk-fix", action="store_true", help="run without GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM "
                    "(llama.cpp treats any value, even 0, as on - so it must be unset)")
    ap.add_argument("extra", nargs="*")
    a = ap.parse_args()
    port = 18123
    args = runtime.server_args(Path(a.model), "Vulkan0", port, a.ctx, False, ram_total_gb=1e9 if a.raw else None)
    args += a.extra
    log = open(Path(__file__).with_name("footprint_server.log"), "ab")
    env = runtime.server_env()
    if a.no_vk_fix:
        env.pop("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", None)
    proc = subprocess.Popen([a.server, *args], env=env, stdout=log, stderr=subprocess.STDOUT)
    url = f"http://127.0.0.1:{port}"
    for _ in range(600):
        try:
            if b'"ok"' in urllib.request.urlopen(url + "/health", timeout=2).read():
                break
        except OSError:
            time.sleep(0.5)
    loaded = proc_stats(proc.pid)
    t0, cpu0 = time.time(), loaded["cpu_s"]
    time.sleep(10)
    idle = proc_stats(proc.pid)
    idle_cpu_pct = 100 * (idle["cpu_s"] - cpu0) / 10 / os.cpu_count()
    msgs, rows = [], []
    for i in range(a.turns):
        msgs.append({"role": "user", "content": f"Turn {i + 1}: tell me three new facts about {TOPICS[i % len(TOPICS)]}."})
        body = {"messages": msgs[-12:], "max_tokens": 160, "temperature": 0.7,
                "chat_template_kwargs": {"enable_thinking": False}}
        req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=600))
        msgs.append({"role": "assistant", "content": d["choices"][0]["message"].get("content") or ""})
        s = proc_stats(proc.pid)
        rows.append({"turn": i + 1, **{k: round(v, 3) for k, v in s.items()},
                     "tok_s": round(d.get("timings", {}).get("predicted_per_second", 0), 1)})
        print(f"turn {i + 1:2}: private {s['private_gb']:.2f} GB  ws {s['working_set_gb']:.2f} GB  "
              f"{rows[-1]['tok_s']} tok/s", flush=True)
    proc.terminate()
    gen_cpu = rows[-1]["cpu_s"] - idle["cpu_s"]
    res = {"label": a.label, "model": Path(a.model).name, "args": args[6:], "loaded": loaded, "idle_cpu_pct": idle_cpu_pct,
           "final": rows[-1], "max_private_gb": max(r["private_gb"] for r in rows),
           "cpu_s_per_turn": gen_cpu / a.turns, "turns": rows}
    with open(Path(__file__).with_name("footprint_results.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")
    print(f"== {a.label}: max private {res['max_private_gb']:.2f} GB, idle CPU {idle_cpu_pct:.2f}%, "
          f"CPU {res['cpu_s_per_turn']:.2f} s/turn")


if __name__ == "__main__":
    main()
