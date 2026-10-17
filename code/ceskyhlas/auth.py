"""Bind address and token rule of the TTS server (standard library only).

TTS_BIND (default 127.0.0.1) is the listen address. On a loopback address (127.0.0.1, ::1, localhost) no
token is needed. On anything else (e.g. 0.0.0.0 for another machine in the LAN) TTS_TOKEN must be set
(the server refuses to start without it) and every request must carry it, either as
"Authorization: Bearer <token>" or as "X-TTS-Token: <token>"; otherwise 401.
"""
from __future__ import annotations

import hmac

LOOPBACK = {"127.0.0.1", "::1", "localhost"}


def token_required(bind: str) -> bool:
    return bind.strip().lower() not in LOOPBACK


def check_config(bind: str, token: str | None) -> str | None:
    """The token to enforce (None = none), or raise SystemExit when the setup is unsafe."""
    token = (token or "").strip() or None
    if token_required(bind) and not token:
        raise SystemExit(f"TTS_BIND={bind} is not a loopback address: set TTS_TOKEN=<secret> (every request "
                         "must then send it), or keep the default TTS_BIND=127.0.0.1")
    if token is not None and len(token) < 16:
        raise SystemExit("TTS_TOKEN must be at least 16 characters (e.g. openssl rand -hex 24)")
    return token


def authorized(headers: dict[str, str], token: str | None) -> bool:
    """headers: lower-case names -> values."""
    if token is None:
        return True
    given = headers.get("x-tts-token", "")
    auth = headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        given = auth[7:].strip()
    return hmac.compare_digest(given.encode(), token.encode())


class TokenAuth:
    """ASGI middleware: 401 for every HTTP request without the token (no-op when token is None)."""

    def __init__(self, app, token: str | None) -> None:
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and self.token is not None:
            headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
            if not authorized(headers, self.token):
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"), (b"www-authenticate", b"Bearer")]})
                await send({"type": "http.response.body", "body": b'{"detail":"missing or wrong TTS token"}'})
                return
        await self.app(scope, receive, send)
