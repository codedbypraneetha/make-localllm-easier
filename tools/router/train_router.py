"""Embed the router data with e5-small (llama.cpp), train a logistic-regression head, evaluate vs the keyword rules.
usage: train_router.py MODEL.gguf [--ngl 0|99]"""
import json, subprocess, sys, time, urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, r"C:\Users\user\Projects\make-localllm-easier")
from localllm import router  # noqa: E402

D = Path(__file__).parent
SERVER = r"C:\Users\user\tools\llama.cpp-b11457\vulkan\llama-server.exe"
LABELS = ["general", "math", "code", "translate"]
MODEL = sys.argv[1]
NGL = sys.argv[sys.argv.index("--ngl") + 1] if "--ngl" in sys.argv else "0"
PORT = 8099


def load(name):
    return [json.loads(l) for l in open(D / "data" / f"{name}.jsonl", encoding="utf-8")]


def post(texts):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/embeddings", data=json.dumps({"input": texts}).encode(),
                                 headers={"Content-Type": "application/json"})
    return [d["embedding"] for d in json.load(urllib.request.urlopen(req, timeout=600))["data"]]


def embed(recs, tag):
    cache = D / "data" / f"emb-{tag}-{Path(MODEL).stem}.npy"
    if cache.exists():
        return np.load(cache)
    out = []
    t = time.time()
    for i in range(0, len(recs), 32):
        out += post(["query: " + r["text"][:450] for r in recs[i:i + 32]])
    print(f"  embedded {len(recs)} {tag} texts in {time.time() - t:.1f}s")
    x = np.array(out, dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    np.save(cache, x)
    return x


def report(name, y, pred, recs):
    acc = 100 * np.mean(np.array(y) == np.array(pred))
    per = {l: round(100 * np.mean([p == l for t, p in zip(y, pred) if t == l]), 1) for l in LABELS}
    bylang = defaultdict(list)
    for t, p, r in zip(y, pred, recs):
        bylang[r["lang"]].append(t == p)
    worst = sorted(((round(100 * np.mean(v), 1), k, len(v)) for k, v in bylang.items()))[:4]
    print(f"  {name:28} {acc:5.1f}%  per class {per}  weakest langs {worst}")
    return acc


srv = subprocess.Popen([SERVER, "-m", MODEL, "--embedding", "--pooling", "mean", "-ngl", NGL, "--port", str(PORT),
                        "-c", "8192", "-b", "8192", "-ub", "8192", "-np", "16"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
try:
    for _ in range(300):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=1); break
        except OSError:
            time.sleep(0.2)
    train, held, test = load("train"), load("heldout"), load("shared_test")
    xtr, xho, xte = embed(train, "train"), embed(held, "heldout"), embed(test, "test")
    ytr, yho, yte = [r["label"] for r in train], [r["label"] for r in held], [r["label"] for r in test]

    best = None
    for c in (0.5, 2, 8, 32):
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced").fit(xtr, ytr)
        a = np.mean(clf.predict(xho) == np.array(yho))
        print(f"  C={c}: held-out {100 * a:.1f}%")
        if best is None or a > best[0]:
            best = (a, c, clf)
    _, c, clf = best
    print(f"\nC={c}")
    report("held-out: embedding router", yho, clf.predict(xho), held)
    report("held-out: keyword rules", yho, [router.detect_task(r["text"]) for r in held], held)
    report("shared test: embedding router", yte, clf.predict(xte), test)
    report("shared test: keyword rules", yte, [router.detect_task(r["text"]) for r in test], test)
    wrong = [(r["lang"], r["label"], p, r["text"][:60]) for r, p in zip(test, clf.predict(xte)) if p != r["label"]]
    print("\nshared-test mistakes:", *wrong, sep="\n  ")

    # single-message latency (what a user feels): embed one text + classify
    lat = []
    for r in test[:40]:
        t = time.perf_counter()
        e = np.array(post(["query: " + r["text"][:450]])[0]); e /= np.linalg.norm(e)
        clf.predict(e[None])
        lat.append(1000 * (time.perf_counter() - t))
    print(f"\nlatency per message (ngl={NGL}): median {np.median(lat):.1f} ms, p95 {np.percentile(lat, 95):.1f} ms")
    json.dump({"labels": list(clf.classes_), "coef": clf.coef_.tolist(), "intercept": clf.intercept_.tolist(),
               "model": Path(MODEL).name, "prefix": "query: "}, open(D / "router_head.json", "w"))
finally:
    srv.terminate()
