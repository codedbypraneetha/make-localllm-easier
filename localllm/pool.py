"""Lazy-load model pool (0.6): several models, one GPU. Only one model is loaded at a time; when the router picks
another one for a message, the loaded llama-server is stopped and the new one started (the file usually comes back
from the OS page cache). Requests are served one at a time (llama-server runs with a single slot anyway), so a swap
never cuts off an answer in progress. Every swap is timed and reported in the X-Localllm-Model header.
"""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager

from . import router


class Pool:
    def __init__(self, keys: list[str], launch, vram_gb: float, ram_free_gb: float = 0.0, first: str | None = None):
        """launch(key) -> (Popen, url) starts a llama-server for `key` and returns once it is healthy."""
        self.keys, self.launch, self.vram_gb, self.ram_free_gb = keys, launch, vram_gb, ram_free_gb
        self.lock = threading.Lock()
        self.proc = self.url = self.current = None
        self.swaps: list[float] = []
        self._swap(first or keys[0])

    def _swap(self, key: str) -> None:
        t = time.time()
        if self.proc is not None:
            self.proc.terminate()
            self.proc.wait(30)
        self.proc, self.url = self.launch(key)
        self.current = key
        self.swaps.append(round(time.time() - t, 2))

    @contextmanager
    def use(self, body: dict):
        """Hold the GPU for one request: pick the model for it, swap if needed, yield (url, header label)."""
        with self.lock:
            key, why = router.pick_local(router._last_user(body), self.keys, self.current, self.vram_gb,
                                         self.ram_free_gb)
            label = f"{key} ({why})"
            if key != self.current:
                self._swap(key)
                label += f", swapped in {self.swaps[-1]:.1f}s"
            yield self.url, label

    def close(self) -> None:
        if self.proc is not None:
            self.proc.terminate()
