"""Fetch an upstream llama.cpp build, list GPUs, and launch llama-server with the settings that measured fastest.

Tuning (each measured on an RX 9070 XT, see README):
  GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM=1  1.7x decode when Resizable BAR is off (llama.cpp#27097); harmless when on
  -np 1 -kvu                             single user: one slot, unified KV
  --spec-type draft-mtp --spec-draft-n-max 2   models that ship an MTP head (Qwen3.8): +40% decode
  -fit off, --load-mode none             skip the fit dry-run (~0.6 s) and read weights straight into VRAM
"""
from __future__ import annotations

import io
import json
import os
import platform
import re
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HOME = Path(os.environ.get("LOCALLLM_HOME", Path.home() / ".localllm"))
EXE = "llama-server.exe" if os.name == "nt" else "llama-server"


def _asset() -> str:
    system = {"Windows": "win", "Linux": "ubuntu", "Darwin": "macos"}[platform.system()]
    arch = "arm64" if platform.machine().lower() in ("arm64", "aarch64") else "x64"
    if system == "macos":
        return f"bin-macos-{arch}"
    return f"bin-{system}-vulkan-{arch}"


def find_server() -> Path:
    if os.environ.get("LOCALLLM_LLAMA_SERVER"):
        return Path(os.environ["LOCALLLM_LLAMA_SERVER"])
    found = sorted((HOME / "llama.cpp").rglob(EXE)) if (HOME / "llama.cpp").exists() else []
    return found[-1] if found else fetch_server()


def fetch_server() -> Path:
    # "latest" can be a source-only tag (v0.6.0 had no binaries); the binaries ride on the per-build bNNNNN releases
    rels = json.load(urllib.request.urlopen("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=10"))
    want = _asset()
    rel, asset = next((r, a) for r in rels for a in r["assets"] if want in a["name"] and a["name"].startswith("llama-"))
    dest = HOME / "llama.cpp" / rel["tag_name"]
    print(f"downloading {asset['name']} ({asset['size'] / 2**20:.0f} MB) ...")
    data = urllib.request.urlopen(asset["browser_download_url"]).read()
    if asset["name"].endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(dest)
    else:
        tarfile.open(fileobj=io.BytesIO(data)).extractall(dest, filter="data")
    exe = next(dest.rglob(EXE))
    if os.name != "nt":
        exe.chmod(0o755)
    return exe


def devices(server: Path) -> list[dict]:
    """[{'id': 'Vulkan0', 'name': ..., 'total_gb': .., 'free_gb': ..}] from `llama-server --list-devices`."""
    out = subprocess.run([str(server), "--list-devices"], capture_output=True, text=True, errors="replace")
    devs = []
    for m in re.finditer(r"^\s*(\w+\d+): (.+?) \((\d+) MiB, (\d+) MiB free\)", out.stdout + out.stderr, re.M):
        devs.append({"id": m[1], "name": m[2], "total_gb": int(m[3]) / 1024, "free_gb": int(m[4]) / 1024})
    return devs


def best_device(devs: list[dict]) -> dict | None:
    """Largest-memory device; integrated GPUs report shared RAM, so prefer names that aren't iGPUs."""
    igpu = re.compile(r"UHD|Iris|Radeon\(TM\) Graphics|Radeon Graphics|890M|780M|Apple", re.I)
    pool = [d for d in devs if not igpu.search(d["name"])] or devs
    return max(pool, key=lambda d: d["total_gb"]) if pool else None


def _windows_memory() -> tuple[float, float]:
    """(total_gb, avail_gb) from GlobalMemoryStatusEx."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                    ("avail", ctypes.c_ulonglong), ("a", ctypes.c_ulonglong), ("b", ctypes.c_ulonglong),
                    ("c", ctypes.c_ulonglong), ("d", ctypes.c_ulonglong), ("e", ctypes.c_ulonglong)]
    s = MS(); s.len = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
    return s.total / 2**30, s.avail / 2**30


def ram_gb() -> float:
    if os.name == "nt":
        return _windows_memory()[0]
    if Path("/proc/meminfo").exists():
        return int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text())[1]) / 2**20
    return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30


def ram_available_gb() -> float:
    """Best-effort currently-available system RAM in GB; falls back to total RAM when unknown."""
    try:
        if os.name == "nt":
            return _windows_memory()[1]
        if Path("/proc/meminfo").exists():
            m = re.search(r"MemAvailable:\s+(\d+)", Path("/proc/meminfo").read_text())
            if m:
                return int(m[1]) / 2**20
    except OSError:
        pass
    return ram_gb()


# Host-RAM caches llama-server keeps by default (8 GiB prompt cache + 32 checkpoints per slot) can grow to many GB
# (llama.cpp #21690: 0.7 -> 18 GB). One user doesn't need that much: size them from the installed RAM.
#   installed RAM (GB) -> (--cache-ram MiB, --ctx-checkpoints)
LOW_RAM_PROFILE = [(16, (512, 2)), (32, (1024, 4)), (64, (2048, 8))]


def ram_profile(ram_total_gb: float) -> tuple[int, int] | None:
    """(cache-ram MiB, ctx-checkpoints) for this PC, or None to keep llama.cpp's defaults on big-RAM machines."""
    for limit, prof in LOW_RAM_PROFILE:
        if ram_total_gb <= limit + 0.5:
            return prof
    return None


def server_args(model: Path, device: str | None, port: int, ctx: int, mtp: bool,
                ram_total_gb: float | None = None, cpu_moe: int = 0) -> list[str]:
    args = ["-m", str(model), "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx), "-ngl", "999",
            "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "-kvu", "-fit", "off"]
    # read weights straight into place. Measured on gemma-4 with 13 layers' experts in RAM: same 36 tok/s as mmap but
    # 6.1 GB working set instead of 13.5 GB (mmap keeps the whole file resident)
    args += ["--load-mode", "none"] + (["--n-cpu-moe", str(cpu_moe)] if cpu_moe else [])
    prof = ram_profile(ram_total_gb if ram_total_gb is not None else ram_gb())
    if prof:
        args += ["--cache-ram", str(prof[0]), "--ctx-checkpoints", str(prof[1])]
    if device:
        args += ["-dev", device]
    if mtp:
        args += ["--spec-type", "draft-mtp", "--spec-draft-n-max", "2"]
    return args


def gpu_spill_gb(pid: int) -> float | None:
    """Windows only: GB of this process's GPU allocations that live in *shared* (system) memory.

    When a model doesn't really fit, WDDM quietly backs part of it with system RAM and decode slows down several
    times; the "GPU Process Memory\\Shared Usage" counter shows it. None when the counter isn't available.
    """
    if os.name != "nt":
        return None
    ps = (f"(Get-Counter '\\GPU Process Memory(pid_{pid}_*)\\Shared Usage' -ErrorAction SilentlyContinue)"
          ".CounterSamples | Measure-Object CookedValue -Sum | ForEach-Object Sum")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip() or 0) / 2**30
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


SPILL_WARN_GB = 1.5   # Vulkan keeps ~0.5-0.9 GB of host-visible buffers here even when the model fits (measured)


def server_env() -> dict:
    return {**os.environ, "GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM": os.environ.get("GGML_VK_DISABLE_HOST_VISIBLE_VIDMEM", "1")}
