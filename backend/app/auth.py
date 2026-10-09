"""Login, roles and signed session tokens, with the standard library only (hashlib, hmac, secrets).

Users live in config/users.json (git-ignored) with PBKDF2-SHA256 password hashes; add them with
`python -m tools.create_user`. Two roles:
    admin    Control Room Admin: everything, including video upload, test cameras and reports
    officer  Security Officer: the command centre and acknowledging alerts

A session token is  base64url(JSON {sub, name, role, iat, exp, jti}) "." base64url(HMAC-SHA256 of
that part with SECRET_KEY), valid for SESSION_HOURS (8). Tokens are stateless; logging out revokes
the token's id until the backend restarts. The API takes the token as "Authorization: Bearer <token>",
or as ?token=<token> where a header cannot be set (<img> and download links); the Socket.IO
connection takes it as auth {token}.
"""

from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import json
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROLES = {"admin": "Control Room Admin", "officer": "Security Officer"}
PBKDF2_ITERATIONS = 240_000
ADMIN_PREFIXES = ("/api/video", "/api/jobs", "/api/reports")  # also /api/videos/...


class TokenError(ValueError):
    """A missing, malformed, tampered, revoked or expired token; the message is shown to the user."""


@dataclass(frozen=True)
class User:
    username: str
    name: str
    role: str

    def public(self) -> dict:
        return {"username": self.username, "name": self.name, "role": self.role, "role_label": ROLES[self.role]}


def hash_password(password: str, iterations: int = PBKDF2_ITERATIONS, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${_b64(salt)}${_b64(digest)}"


def verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iterations, salt, digest = stored.split("$")
        if scheme != "pbkdf2_sha256":
            return False
        expected = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), _unb64(salt), int(iterations))
        return hmac.compare_digest(expected, _unb64(digest))
    except (ValueError, TypeError):
        return False


@functools.lru_cache(maxsize=1)
def _dummy_hash() -> str:
    return hash_password("not a real password", salt=b"0" * 16)


class UserStore:
    """config/users.json, re-read when the file changes (so tools/create_user.py needs no restart)."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._mtime: float | None = None
        self._users: dict[str, tuple[User, str]] = {}
        self._lock = threading.Lock()

    def _load(self) -> dict[str, tuple[User, str]]:
        with self._lock:
            mtime = self.path.stat().st_mtime if self.path.exists() else None
            if mtime != self._mtime:
                users = {}
                if mtime is not None:
                    for raw in json.loads(self.path.read_text(encoding="utf-8")).get("users", []):
                        if raw.get("role") in ROLES:
                            user = User(raw["username"], raw.get("name") or raw["username"], raw["role"])
                            users[user.username.lower()] = (user, raw["password_hash"])
                self._users, self._mtime = users, mtime
            return self._users

    def count(self) -> int:
        return len(self._load())

    def authenticate(self, username: str, password: str) -> User | None:
        entry = self._load().get(username.strip().lower())
        # Hash even for an unknown user, so the response time does not reveal which names exist.
        ok = verify_password(password, entry[1] if entry else _dummy_hash())
        return entry[0] if entry and ok else None


class Sessions:
    """Issues and checks signed session tokens."""

    def __init__(self, secret: str, hours: float = 8.0) -> None:
        if not secret:
            raise ValueError("SECRET_KEY is empty")
        self._key = secret.encode("utf-8")
        self.ttl_s = hours * 3600
        self._revoked: dict[str, float] = {}  # jti -> expiry
        self._lock = threading.Lock()

    def _sign(self, body: str) -> str:
        return _b64(hmac.new(self._key, body.encode("ascii"), hashlib.sha256).digest())

    def issue(self, user: User, now: float | None = None) -> tuple[str, dict]:
        now = time.time() if now is None else now
        claims = {"sub": user.username, "name": user.name, "role": user.role, "iat": int(now),
                  "exp": int(now + self.ttl_s), "jti": secrets.token_hex(8)}
        body = _b64(json.dumps(claims, separators=(",", ":")).encode("utf-8"))
        return f"{body}.{self._sign(body)}", claims

    def verify(self, token: str | None, now: float | None = None) -> dict:
        if not token:
            raise TokenError("Not logged in: please log in.")
        try:
            body, signature = token.split(".")
            if not hmac.compare_digest(signature, self._sign(body)):
                raise ValueError("bad signature")
            claims = json.loads(_unb64(body))
        except (ValueError, TypeError, UnicodeDecodeError):
            raise TokenError("Invalid session: please log in again.") from None
        if claims.get("role") not in ROLES:
            raise TokenError("Invalid session: please log in again.")
        if (time.time() if now is None else now) >= claims.get("exp", 0):
            raise TokenError("Your session has expired: please log in again.")
        with self._lock:
            if claims.get("jti") in self._revoked:
                raise TokenError("You have logged out: please log in again.")
        return claims

    def revoke(self, claims: dict) -> None:
        now = time.time()
        with self._lock:
            self._revoked = {j: exp for j, exp in self._revoked.items() if exp > now}  # forget expired ones
            self._revoked[claims["jti"]] = claims["exp"]


def required_role(method: str, path: str) -> str | None:
    """None for public paths, "any" for a logged-in user, "admin" for admin-only ones."""
    if not path.startswith("/api/") or (method == "POST" and path == "/api/auth/login"):
        return None  # /health, /docs, the dashboard's own files; Socket.IO checks its token itself
    if path.startswith(ADMIN_PREFIXES):
        return "admin"
    if path.startswith("/api/zones/") and (method == "DELETE" or path.endswith("/infer")):
        return "admin"  # removing a test camera; running inference by hand
    return "any"


def user_from_claims(claims: dict) -> User:
    return User(claims["sub"], claims.get("name", claims["sub"]), claims["role"])


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
