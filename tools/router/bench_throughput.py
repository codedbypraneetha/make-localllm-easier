"""Bulk embedding throughput, GPU vs CPU (texts/s), on 2,000 training texts."""
import json, subprocess, time, urllib.request
from pathlib import Path

D = Path(__file__).parent
SERVER = r"C:\Users\user\tools\llama.cpp-b11457\vulkan\llama-server.exe"
texts = ["query: " + json.loads(l)["text"][:450] for l in open(D / "data" / "train.jsonl", encoding="utf-8")][:2000]
for ngl in ("99", "0"):
    srv = subprocess.Popen([SERVER, "-m", str(D / "e5-small-q8_0.gguf"), "--embedding", "--pooling", "mean", "-ngl", ngl,
                            "--port", "8098", "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "16"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(300):
            try:
                urllib.request.urlopen("http://127.0.0.1:8098/health", timeout=1); break
            except OSError:
                time.sleep(0.2)
        def post(b):
            req = urllib.request.Request("http://127.0.0.1:8098/v1/embeddings", data=json.dumps({"input": b}).encode(),
                                         headers={"Content-Type": "application/json"})
            return json.load(urllib.request.urlopen(req, timeout=600))
        post(texts[:32])
        t = time.time()
        for i in range(0, len(texts), 64):
            try:
                post(texts[i:i + 64])
            except Exception:
                pass
        print(f"ngl={ngl}: {len(texts) / (time.time() - t):.0f} texts/s")
    finally:
        srv.terminate(); srv.wait(10)
