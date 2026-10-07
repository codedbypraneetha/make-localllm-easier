"""Local-first routing. Every chat request stays on this PC unless the user turned routing on in
~/.localllm/route.json AND a rule matches. Keys are never stored by localllm: they come from environment variables (or
the OS keychain via the optional `keyring` package) named in the config.

Example ~/.localllm/route.json
{
  "enabled": true,
  "provider": "openrouter",
  "providers": {
    "openrouter": {"kind": "openai", "url": "https://openrouter.ai/api", "key_env": "OPENROUTER_API_KEY",
                   "model": "anthropic/claude-sonnet-4"},
    "anthropic":  {"kind": "anthropic", "url": "https://api.anthropic.com", "key_env": "ANTHROPIC_API_KEY",
                   "model": "claude-sonnet-4-5"}
  },
  "rules": {"max_local_prompt_tokens": 6000, "language_floor": 60, "cloud_model_names": true},
  "local_model": "qwen3.8-27b-q3"
}
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass

from .runtime import HOME

CONFIG = HOME / "route.json"
SCRIPTS = [("th", r"[฀-๿]"), ("hi", r"[ऀ-ॿ]"), ("ar", r"[؀-ۿ]"), ("ja", r"[぀-ヿ]"),
           ("ko", r"[가-힯]"), ("zh", r"[一-鿿]"), ("ru", r"[Ѐ-ӿ]"), ("he", r"[֐-׿]"),
           ("el", r"[Ͱ-Ͽ]"), ("bn", r"[ঀ-৿]"), ("ta", r"[஀-௿]")]
CLOUD_NAMES = re.compile(r"^(gpt-|o\d|claude-|gemini-|anthropic/|openai/|google/)", re.I)


def detect_language(text: str) -> str:
    """Cheap script-based guess (no model): enough to look up per-language scores."""
    counts = {lang: len(re.findall(rx, text)) for lang, rx in SCRIPTS}
    lang, n = max(counts.items(), key=lambda kv: kv[1])
    return lang if n >= 3 else "en"


@dataclass
class Provider:
    name: str
    kind: str          # "openai" (OpenAI-compatible) or "anthropic"
    url_base: str
    key: str
    model: str

    def serves(self, path: str) -> bool:
        return (self.kind == "openai" and path == "/v1/chat/completions") or \
               (self.kind == "anthropic" and path == "/v1/messages")

    def headers(self, path: str) -> dict:
        if self.kind == "anthropic":
            return {"x-api-key": self.key, "anthropic-version": "2023-06-01"}
        return {"Authorization": f"Bearer {self.key}"}

    def adapt(self, body: bytes, path: str) -> bytes:
        d = json.loads(body or b"{}")
        d["model"] = self.model
        return json.dumps(d).encode()


@dataclass
class Decision:
    label: str                      # shown to the user in X-Localllm-Route
    provider: Provider | None = None


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"enabled": False}


def _key(env_name: str) -> str | None:
    if os.environ.get(env_name):
        return os.environ[env_name]
    try:
        import keyring  # optional
        return keyring.get_password("localllm", env_name)
    except Exception:
        return None


def providers(cfg: dict) -> dict[str, Provider]:
    out = {}
    for name, p in (cfg.get("providers") or {}).items():
        key = _key(p.get("key_env", ""))
        if key:
            out[name] = Provider(name, p.get("kind", "openai"), p["url"], key, p.get("model", ""))
    return out


def _text(body: dict) -> str:
    parts = []
    for m in body.get("messages", []):
        c = m.get("content")
        parts.append(c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict)))
    if isinstance(body.get("system"), str):
        parts.append(body["system"])
    return "\n".join(parts)


def _last_user(body: dict) -> str:
    for m in reversed(body.get("messages", [])):
        if m.get("role") == "user":
            c = m.get("content")
            return c if isinstance(c, str) else " ".join(x.get("text", "") for x in c or [] if isinstance(x, dict))
    return ""


def decide(body: dict, path: str, cfg: dict | None = None) -> Decision:
    cfg = load_config() if cfg is None else cfg
    if not cfg.get("enabled"):
        return Decision("local")
    provs = providers(cfg)
    usable = [p for p in provs.values() if p.serves(path)]
    preferred = provs.get(cfg.get("provider", ""))
    cloud = preferred if preferred and preferred.serves(path) else (usable[0] if usable else None)
    if not cloud:
        return Decision("local (no cloud key configured for this API)")
    rules = cfg.get("rules") or {}
    if rules.get("cloud_model_names", True) and CLOUD_NAMES.match(str(body.get("model", ""))):
        return Decision(f"cloud:{cloud.name} (app asked for {body['model']})", cloud)
    tokens = len(_text(body)) // 3
    if tokens > rules.get("max_local_prompt_tokens", 10 ** 9):
        return Decision(f"cloud:{cloud.name} (prompt ~{tokens} tokens > local limit)", cloud)
    floor = rules.get("language_floor")
    if floor:
        from . import catalog
        lang = detect_language(_last_user(body))
        local = cfg.get("local_model")
        scores = catalog.MODELS.get(local, {}).get("scores", {}) if local else {}
        mine = [v for k, v in scores.items() if k.startswith(lang + "/")]
        if mine and sum(mine) / len(mine) < floor:
            return Decision(f"cloud:{cloud.name} ({lang} score {sum(mine) / len(mine):.0f} < floor {floor})", cloud)
    return Decision("local")
