"""vibe.py: key aus keyring/dotenv, cache, request-hook (headers, reasoning, effort).

braucht nur pytest, kein pyrate:  ~/git/osuv/pyrate/.venv/bin/python -m pytest -q ~/git/osuv/pyrate-setup
"""

import copy
import subprocess
from types import SimpleNamespace

import pytest

import vibe  # liegt daneben, pytest packt das verzeichnis in sys.path


@pytest.fixture(autouse=True)
def _clean(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBE_HOME", str(tmp_path / "vibe"))
    vibe.invalidate()
    yield
    vibe.invalidate()


def fake_run(monkeypatch, results):
    """subprocess.run-ersatz: liefert pro aufruf das nächste (returncode, stdout) bzw. wirft es."""
    calls = []

    def run(argv, **kw):
        calls.append(argv)
        r = results[len(calls) - 1]
        if isinstance(r, Exception):
            raise r
        return subprocess.CompletedProcess(argv, r[0], stdout=r[1])

    monkeypatch.setattr(vibe.subprocess, "run", run)
    return calls


def test_dotenv(tmp_path):
    (tmp_path / "vibe").mkdir()
    env = tmp_path / "vibe" / ".env"
    env.write_text("FOO=bar\n# comment\nexport MISTRAL_API_KEY='vsk-geheim'\n", encoding="utf-8")
    assert vibe._read_dotenv() == "vsk-geheim"
    env.write_text("FOO=bar\n", encoding="utf-8")
    assert vibe._read_dotenv() is None


def test_secret_tool_falls_back_to_legacy_service(monkeypatch):
    monkeypatch.setattr(vibe.sys, "platform", "linux")
    calls = fake_run(monkeypatch, [(1, ""), (0, "vsk-legacy\n")])
    assert vibe._read_os_keyring() == "vsk-legacy"
    assert calls == [["secret-tool", "lookup", "service", s, "username", "MISTRAL_API_KEY"]
                     for s in ("ai.mistral.vibe", "vibe")]


def test_missing_secret_tool_tries_once(monkeypatch):
    monkeypatch.setattr(vibe.sys, "platform", "linux")
    calls = fake_run(monkeypatch, [FileNotFoundError("secret-tool")])
    assert vibe._read_os_keyring() is None
    assert len(calls) == 1


def test_cache_includes_missing_key(monkeypatch):
    calls = []
    monkeypatch.setattr(vibe, "_read", lambda: calls.append(1))  # → None
    assert vibe.key() is None and vibe.key() is None
    assert len(calls) == 1  # auch "kein key" ist gecacht, sonst subprozesse pro request
    monkeypatch.setattr(vibe, "TTL", 0)
    vibe.key()
    assert len(calls) == 2


def call(api_key="", session=None, **payload):
    """plugin-hook direkt aufrufen → (headers, payload)."""
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    payload.setdefault("messages", [])
    vibe.request(SimpleNamespace(api_key=api_key), headers, payload, session)
    return headers, payload


def test_headers(monkeypatch):
    monkeypatch.setattr(vibe, "_read", lambda: "vsk-geheim")
    assert call(session="sess-1")[0] == {"Authorization": "Bearer vsk-geheim", "X-Affinity": "sess-1"}


def test_explicit_key_wins(monkeypatch):
    monkeypatch.setattr(vibe, "_read", lambda: "vsk-geheim")
    assert call(api_key="sk-explicit")[0] == {"Authorization": "Bearer sk-explicit"}


def test_missing_key(monkeypatch):
    monkeypatch.setattr(vibe, "_read", lambda: None)
    with pytest.raises(RuntimeError, match="vibe --setup"):
        call()


def test_reasoning_becomes_thinking_chunk():
    """mistral will reasoning als thinking-chunk im content, reasoning_content gibt 422."""
    tc = [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]
    msgs = [
        {"role": "user", "content": "los"},
        {"role": "assistant", "content": None, "tool_calls": tc, "reasoning_content": "hmm"},
        {"role": "tool", "tool_call_id": "c1", "content": "inhalt"},
        {"role": "assistant", "content": "fertig", "reasoning_content": "aha",
         "thinking_blocks": [{"type": "thinking", "thinking": "aha", "signature": "x"}]},
        {"role": "assistant", "content": "ohne"},
    ]
    before = copy.deepcopy(msgs)
    out = call(api_key="k", messages=msgs)[1]["messages"]
    think = lambda t: {"type": "thinking", "thinking": [{"type": "text", "text": t}]}
    assert out[1] == {"role": "assistant", "content": [think("hmm")], "tool_calls": tc}
    assert out[3] == {"role": "assistant", "content": [think("aha"), {"type": "text", "text": "fertig"}]}
    assert out[0] is msgs[0] and out[2] is msgs[2] and out[4] is msgs[4]
    assert msgs == before  # die messages gehören dem agent


@pytest.mark.parametrize(("effort", "sent"), [("low", "none"), ("medium", "high"), ("max", "high")])
def test_effort(effort, sent):
    assert call(api_key="k", reasoning_effort=effort)[1]["reasoning_effort"] == sent


def test_no_effort_stays_unset():
    assert "reasoning_effort" not in call(api_key="k")[1]

