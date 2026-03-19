from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
from typing import Mapping

from .util import utc_now_unix


def signing_secret(*, configured_secret: str, api_token: str) -> str:
    return (configured_secret or api_token or "").strip()


def build_signature(*, path: str, expires_at: int, secret: str) -> str:
    payload = f"{path}\n{int(expires_at)}".encode("utf-8")
    digest = hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def create_signed_query(*, path: str, ttl_seconds: int, secret: str) -> dict[str, str]:
    expires_at = utc_now_unix() + max(1, int(ttl_seconds))
    return {
        "exp": str(expires_at),
        "sig": build_signature(path=path, expires_at=expires_at, secret=secret),
    }


def has_valid_signature(*, path: str, query_params: Mapping[str, object], secret: str) -> bool:
    secret = (secret or "").strip()
    if not secret:
        return False

    sig = str(query_params.get("sig") or "").strip()
    exp_raw = str(query_params.get("exp") or "").strip()
    if not sig or not exp_raw:
        return False

    try:
        expires_at = int(exp_raw)
    except ValueError:
        return False
    if expires_at < utc_now_unix():
        return False

    expected = build_signature(path=path, expires_at=expires_at, secret=secret)
    return secrets.compare_digest(sig, expected)
