"""Run model-written code safely: inside a throwaway Docker container, never on the host.

The container has no network, a read-only filesystem (only /tmp is writable), no Linux capabilities, no privilege
escalation, a memory/CPU/process cap, a non-root user, and a per-program timeout. Programs are passed in through a
read-only mount and results come back as JSON on stdout. If Docker isn't available the code is NOT run - there is no
unsafe fallback.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

IMAGE = "localllm-sandbox:1"
DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir numpy==2.1.3 && useradd -m -u 10001 runner
USER runner
WORKDIR /tmp
"""
RUNNER = r'''
import json, os, subprocess, sys
res = []
for name in sorted(os.listdir("/work")):
    if not (name.startswith("p") and name.endswith(".py")):
        continue
    try:
        p = subprocess.run([sys.executable, "/work/" + name], capture_output=True, timeout=float(sys.argv[1]), cwd="/tmp")
        res.append({"name": name, "ok": p.returncode == 0, "err": p.stderr.decode("utf-8", "replace")[-300:]})
    except subprocess.TimeoutExpired:
        res.append({"name": name, "ok": False, "err": "timeout"})
print(json.dumps(res))
'''


def docker() -> str | None:
    exe = shutil.which("docker")
    if not exe:
        return None
    try:
        ok = subprocess.run([exe, "info"], capture_output=True, timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return None
    return exe if ok else None


def ensure_image(exe: str) -> None:
    if subprocess.run([exe, "image", "inspect", IMAGE], capture_output=True).returncode == 0:
        return
    subprocess.run([exe, "build", "-t", IMAGE, "-"], input=DOCKERFILE.encode(), check=True, capture_output=True,
                   timeout=900)


def command(exe: str, work: str, timeout_s: float) -> list[str]:
    return [exe, "run", "--rm", "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--memory", "1g", "--memory-swap", "1g",
            "--cpus", "2", "--pids-limit", "256", "--user", "10001", "-v", f"{work}:/work:ro", IMAGE,
            "python", "/work/_runner.py", str(timeout_s)]


def run(programs: list[str], timeout_s: float = 10.0) -> list[dict]:
    """Run each program in the sandbox; [{'ok': bool, 'err': str}] in input order. Raises if Docker is missing."""
    exe = docker()
    if not exe:
        raise RuntimeError("Docker is not running: model-written code is only executed inside a container")
    ensure_image(exe)
    with tempfile.TemporaryDirectory(prefix="localllm-sbx-") as d:
        for i, src in enumerate(programs):
            Path(d, f"p{i:05d}.py").write_text(src, encoding="utf-8")
        Path(d, "_runner.py").write_text(RUNNER, encoding="utf-8")
        out = subprocess.run(command(exe, d, timeout_s), capture_output=True, text=True,
                             timeout=60 + timeout_s * len(programs))
        rows = {r["name"]: r for r in json.loads(out.stdout.strip().splitlines()[-1])}
    return [rows.get(f"p{i:05d}.py", {"ok": False, "err": "missing"}) for i in range(len(programs))]
