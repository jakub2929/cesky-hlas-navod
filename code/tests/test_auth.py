"""ceskyhlas/auth.py: loopback without a token, any other bind only with a token on every request."""
import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ceskyhlas.auth import TokenAuth, authorized, check_config, token_required  # noqa: E402

TOKEN = "test-token-not-a-secret"


def test_loopback_needs_no_token():
    for b in ("127.0.0.1", "::1", "localhost"):
        assert not token_required(b)
        assert check_config(b, None) is None


def test_open_bind_requires_a_token():
    for b in ("0.0.0.0", "lan-host.local", "::"):
        assert token_required(b)
        with pytest.raises(SystemExit):
            check_config(b, None)
        with pytest.raises(SystemExit):
            check_config(b, "  ")
    with pytest.raises(SystemExit):
        check_config("0.0.0.0", "short")
    assert check_config("0.0.0.0", TOKEN) == TOKEN


def test_authorized_headers():
    assert authorized({}, None)
    assert authorized({"authorization": f"Bearer {TOKEN}"}, TOKEN)
    assert authorized({"x-tts-token": TOKEN}, TOKEN)
    assert not authorized({}, TOKEN)
    assert not authorized({"authorization": "Bearer wrong-token-0000000"}, TOKEN)
    assert not authorized({"x-tts-token": TOKEN[:-1]}, TOKEN)


def _call(mw, headers):
    sent, reached = [], []

    async def app(scope, receive, send):
        reached.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})

    async def receive():
        return {"type": "http.request", "body": b""}

    async def send(msg):
        sent.append(msg)

    scope = {"type": "http", "headers": [(k.encode(), v.encode()) for k, v in headers.items()]}
    asyncio.run(TokenAuth(app, mw)(scope, receive, send))
    return sent[0]["status"], bool(reached)


def test_middleware():
    assert _call(None, {}) == (200, True)                                   # loopback: open
    assert _call(TOKEN, {}) == (401, False)                                 # token missing
    assert _call(TOKEN, {"Authorization": "Bearer nope-nope-nope-nope"}) == (401, False)
    assert _call(TOKEN, {"Authorization": f"Bearer {TOKEN}"}) == (200, True)
    assert _call(TOKEN, {"X-TTS-Token": TOKEN}) == (200, True)
