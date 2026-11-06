"""pyrate-plugin für mistral vibe: nimmt den api-key, den `vibe --setup` abgelegt hat,
und biegt die stellen gerade, an denen api.mistral.ai vom openai-format abweicht.

port des opencode-plugins github.com/thorstenkramm/opencode-mistral-vibe. nur stdlib,
importiert nichts aus pyrate.

- key-quelle: macos keychain bzw. linux secret-service (per `secret-tool`, paket
  libsecret-tools), sonst $VIBE_HOME/.env (default ~/.vibe/.env). max. 1×/minute neu gelesen
- ein `api_key` am backend gewinnt – das plugin taugt also auch für mistral ohne vibe
- schickt die session-id als X-Affinity mit (prompt-cache, wie vibe selbst)
- reasoning im tool-loop geht als thinking-chunk zurück statt als reasoning_content (sonst 422)
- reasoning_effort: low → none, alles andere → high (mehr kennt mistral nicht; glm nur high)
- overflow-recovery braucht nichts: pyrate erkennt mistrals "maximum context length" schon

    backends:
      - name: vibe
        base_url: https://api.mistral.ai/v1
        plugin: ~/git/osuv/pyrate-setup/plugins/vibe.py
        context_limit: 262144   # sonst keine proaktive auto-compaction
        models: [mistral-vibe-cli-latest, mistral-vibe-cli-with-tools, mistral-vibe-cli-fast]

modelle sind aliases: cli-latest und cli-with-tools → medium, cli-fast → small 4. ob sie
übers vibe-abo oder als normale api-nutzung abgerechnet werden: in der mistral-console prüfen.
"""

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

KEY_NAME = "MISTRAL_API_KEY"
SERVICES = ("ai.mistral.vibe", "vibe")  # aktueller service, dann legacy
TTL = 60       # s; `vibe --setup` neu ausgeführt → ohne restart übernehmen
TIMEOUT = 60   # s; zeit zum entsperren des keyrings (wie vibe selbst)

_lock = threading.Lock()  # sonst poppen bei parallelen requests mehrere entsperr-dialoge auf
_cache = None             # (zeitpunkt, key); key darf None sein, wird auch gecacht


def _lookup_cmd(service):
    if sys.platform == "darwin":
        return ["security", "find-generic-password", "-s", service, "-a", KEY_NAME, "-w"]
    # python-keyring (das vibe nutzt) speichert unter den attributen service + username
    return ["secret-tool", "lookup", "service", service, "username", KEY_NAME]


def _read_os_keyring():
    for service in SERVICES:
        try:
            out = subprocess.run(_lookup_cmd(service), capture_output=True, timeout=TIMEOUT, text=True, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return None  # tool fehlt oder keyring bleibt zu – nächster service hilft da nicht
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    return None


def _read_dotenv():
    home = Path(os.environ.get("VIBE_HOME") or Path.home() / ".vibe")
    try:
        content = (home / ".env").read_text(encoding="utf-8")
    except OSError:
        return None
    prefix = f"{KEY_NAME}="
    for line in content.splitlines():
        line = line.strip().removeprefix("export ")
        if not line.startswith(prefix):
            continue
        value = line[len(prefix):].strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if value:
            return value
    return None


def _read():
    return _read_os_keyring() or _read_dotenv()


def key():
    """vibe api-key oder None, maximal TTL sekunden gecacht."""
    global _cache
    with _lock:
        if _cache is None or time.monotonic() - _cache[0] >= TTL:
            value = _read()
            _cache = (time.monotonic(), value)
        return _cache[1]


def invalidate():
    global _cache
    with _lock:
        _cache = None


REASONING_KEYS = ("reasoning_content", "thinking_blocks")


def _message(m):
    """reasoning wie vibe als thinking-chunk in den content – mistral lehnt reasoning_content
    am assistant-message mit 422 ab. gibt eine kopie zurück, das original gehört pyrate."""
    if m.get("role") != "assistant" or not any(k in m for k in REASONING_KEYS):
        return m
    out = {k: v for k, v in m.items() if k not in REASONING_KEYS}
    if m.get("reasoning_content"):
        out["content"] = [{"type": "thinking", "thinking": [{"type": "text", "text": m["reasoning_content"]}]}]
        if m.get("content"):
            out["content"].append({"type": "text", "text": m["content"]})
    return out


def request(backend, headers, payload, session):
    """pyrate-hook, einmal vor jedem request."""
    if session:
        headers["X-Affinity"] = session
    if not backend.api_key:
        k = key()
        if not k:
            raise RuntimeError("kein api-key gefunden – `vibe --setup` ausführen. unter linux braucht's "
                               "außerdem `secret-tool` (libsecret-tools) und einen entsperrten keyring")
        headers["Authorization"] = f"Bearer {k}"
    payload["messages"] = [_message(m) for m in payload["messages"]]
    # mistral kennt nur none/high (glm sogar nur high). mapping wie vibe selbst
    if "reasoning_effort" in payload:
        payload["reasoning_effort"] = "none" if payload["reasoning_effort"] == "low" else "high"
