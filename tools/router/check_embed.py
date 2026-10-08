"""Does the GGUF (llama.cpp, CPU) give the same embeddings as the original PyTorch e5-small? usage: check_embed.py MODEL.gguf"""
import json, subprocess, sys, time, urllib.request
import torch
from transformers import AutoModel, AutoTokenizer

TEXTS = ["query: How many moons does Jupiter have?", "query: ช่วยแปลประโยคนี้เป็นภาษาอังกฤษให้หน่อย",
         "query: 请帮我写一个Python函数计算斐波那契数列", "query: Janet has 16 eggs and eats 3. How many are left?",
         "query: Écris un poème sur la mer.", "query: इस लेख का सारांश लिखिए"]
SERVER = r"C:\Users\user\tools\llama.cpp-b11457\vulkan\llama-server.exe"

tok = AutoTokenizer.from_pretrained("e5-small")
model = AutoModel.from_pretrained("e5-small").eval()
with torch.no_grad():
    b = tok(TEXTS, padding=True, return_tensors="pt")
    h = model(**b).last_hidden_state
    m = b["attention_mask"].unsqueeze(-1).float()
    ref = torch.nn.functional.normalize((h * m).sum(1) / m.sum(1), dim=-1)

srv = subprocess.Popen([SERVER, "-m", sys.argv[1], "--embedding", "--pooling", "mean", "-ngl", "0", "--port", "8099",
                        "-c", "512", "-b", "512", "-ub", "512"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(200):
        try:
            urllib.request.urlopen("http://127.0.0.1:8099/health", timeout=1); break
        except OSError:
            time.sleep(0.2)
    req = urllib.request.Request("http://127.0.0.1:8099/v1/embeddings", data=json.dumps({"input": TEXTS}).encode(),
                                 headers={"Content-Type": "application/json"})
    got = torch.tensor([d["embedding"] for d in json.load(urllib.request.urlopen(req))["data"]])
    got = torch.nn.functional.normalize(got, dim=-1)
    cos = (got * ref).sum(-1)
    print("cosine vs PyTorch per text:", [round(float(c), 4) for c in cos])
finally:
    srv.terminate()
