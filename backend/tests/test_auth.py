"""P1: login, roles and session tokens (app/auth.py, app/main.py, tools/create_user.py)."""

import asyncio
import logging
from dataclasses import replace

import pytest
import socketio
from fastapi.testclient import TestClient

from app.auth import Sessions, TokenError, User, UserStore, hash_password, required_role, verify_password
from app.main import HideTokens, create_app
from tests.fakes import auth_headers, make_services
from tools.create_user import ensure_secret_key, load, save, upsert

ADMIN_PW, OFFICER_PW = "Admin-Pass-2027", "Officer-Pass-2027"


@pytest.fixture
def app_and_client(tmp_path):
    users = tmp_path / "users.json"
    data = load(users)
    upsert(data, "admin1", "Control Room", "admin", ADMIN_PW)
    upsert(data, "officer1", "Officer One", "officer", OFFICER_PW)
    save(users, data)
    services = make_services(count=100)
    services = replace(services, settings=replace(services.settings, users_config=users))
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    with TestClient(app) as client:
        yield app, client


def login(client, username, password):
    return client.post("/api/auth/login", json={"username": username, "password": password})


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_passwords_are_salted_pbkdf2_hashes():
    a, b = hash_password("secret-123"), hash_password("secret-123")
    assert a != b and a.startswith("pbkdf2_sha256$240000$")
    assert verify_password("secret-123", a) and not verify_password("secret-124", a)
    assert not verify_password("secret-123", "garbage")


def test_login_me_and_the_api_with_the_token(app_and_client):
    _, client = app_and_client
    response = login(client, "Admin1", ADMIN_PW)  # usernames are not case-sensitive
    assert response.status_code == 200
    body = response.json()
    assert body["user"] == {"username": "admin1", "name": "Control Room", "role": "admin", "role_label": "Control Room Admin"}
    me = client.get("/api/auth/me", headers=bearer(body["token"])).json()
    assert me["username"] == "admin1" and me["expires_at"] == body["expires_at"]
    assert client.get("/api/zones", headers=bearer(body["token"])).status_code == 200


def test_wrong_password_and_unknown_user_get_the_same_answer(app_and_client):
    _, client = app_and_client
    for username, password in (("admin1", "wrong-password"), ("nobody", ADMIN_PW)):
        response = login(client, username, password)
        assert response.status_code == 401 and response.json()["detail"] == "Wrong username or password."


def test_the_api_needs_a_valid_unexpired_token(app_and_client):
    app, client = app_and_client
    assert client.get("/health").status_code == 200  # public
    missing = client.get("/api/zones")
    assert missing.status_code == 401 and "log in" in missing.json()["detail"]
    token, _ = app.sessions.issue(User("admin1", "A", "admin"), now=0)  # issued in 1970: long expired
    expired = client.get("/api/zones", headers=bearer(token))
    assert expired.status_code == 401 and "expired" in expired.json()["detail"]
    good = login(client, "admin1", ADMIN_PW).json()["token"]
    tampered = good[:-2] + ("AA" if not good.endswith("AA") else "BB")
    assert client.get("/api/zones", headers=bearer(tampered)).status_code == 401
    other_key = Sessions("another-secret").issue(User("admin1", "A", "admin"))[0]
    assert client.get("/api/zones", headers=bearer(other_key)).status_code == 401
    # <img> and download links cannot send a header: ?token= works too
    assert client.get(f"/api/zones?token={good}").status_code == 200


def test_logout_ends_the_session(app_and_client):
    _, client = app_and_client
    token = login(client, "officer1", OFFICER_PW).json()["token"]
    assert client.post("/api/auth/logout", headers=bearer(token)).status_code == 204
    after = client.get("/api/auth/me", headers=bearer(token))
    assert after.status_code == 401 and "logged out" in after.json()["detail"]


def test_officers_cannot_use_admin_features(app_and_client):
    _, client = app_and_client
    officer = bearer(login(client, "officer1", OFFICER_PW).json()["token"])
    admin = bearer(login(client, "admin1", ADMIN_PW).json()["token"])
    for method, path in (("GET", "/api/video/options"), ("GET", "/api/jobs"), ("GET", "/api/reports/summary"),
                         ("POST", "/api/videos?filename=a.mp4"), ("DELETE", "/api/zones/VIDEO_1"),
                         ("POST", "/api/zones/RAMKUND_NORTH/infer")):
        response = client.request(method, path, headers=officer)
        assert response.status_code == 403 and "Control Room Admin" in response.json()["detail"], path
    for path in ("/api/zones", "/api/alerts/recent", "/api/config", "/api/zones/RAMKUND_NORTH/history"):
        assert client.get(path, headers=officer).status_code == 200, path
    assert client.get("/api/video/options", headers=admin).status_code == 200
    assert client.get("/api/reports/summary", headers=admin).status_code == 200


def test_no_users_set_up_is_explained(tmp_path):
    services = make_services()
    settings = replace(services.settings, users_config=tmp_path / "none.json")
    app = create_app(settings, services_factory=lambda _: replace(services, settings=settings), run_loop=False)
    with TestClient(app) as client:
        response = login(client, "admin1", ADMIN_PW)
    assert response.status_code == 503 and "create_user" in response.json()["detail"]


def test_the_socket_refuses_a_missing_or_bad_token():
    services = make_services()
    app = create_app(services.settings, services_factory=lambda _: services, run_loop=False)
    connect = app.sio.handlers["/"]["connect"]
    for auth in (None, {}, {"token": "nonsense"}):
        with pytest.raises(socketio.exceptions.ConnectionRefusedError):
            asyncio.run(connect("sid1", {}, auth))
    token = auth_headers(services.settings)["Authorization"][7:]
    asyncio.run(connect("sid2", {}, {"token": token}))  # accepted: no exception


def test_route_rules():
    assert required_role("GET", "/health") is None and required_role("POST", "/api/auth/login") is None
    assert required_role("GET", "/api/zones") == "any" and required_role("POST", "/api/alerts/5/ack") == "any"
    assert required_role("POST", "/api/videos") == "admin" and required_role("GET", "/api/reports/alerts") == "admin"
    assert required_role("DELETE", "/api/zones/VIDEO_1") == "admin"


def test_tokens_expire_after_the_session_length():
    sessions = Sessions("k", hours=8)
    token, claims = sessions.issue(User("u", "U", "officer"), now=1000.0)
    assert claims["exp"] == 1000 + 8 * 3600
    assert sessions.verify(token, now=1000 + 8 * 3600 - 1)["sub"] == "u"
    with pytest.raises(TokenError, match="expired"):
        sessions.verify(token, now=1000 + 8 * 3600)


def test_create_user_tool_and_secret_key(tmp_path):
    env = tmp_path / ".env"
    env.write_text("MONGODB_DB=kumbhmela", encoding="utf-8")  # no newline at the end, no SECRET_KEY
    assert ensure_secret_key(env) and not ensure_secret_key(env)  # added once
    lines = env.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "MONGODB_DB=kumbhmela" and lines[-1].startswith("SECRET_KEY=") and len(lines[-1]) > 50
    users = tmp_path / "users.json"
    data = load(users)
    upsert(data, "officer1", "Officer", "officer", "first-password")
    upsert(data, "OFFICER1", "Officer", "officer", "second-password")  # same user: replaced
    save(users, data)
    assert "password" not in users.read_text(encoding="utf-8").replace("password_hash", "")
    store = UserStore(users)
    assert store.count() == 1 and store.authenticate("officer1", "second-password").role == "officer"
    assert store.authenticate("officer1", "first-password") is None


def test_the_access_log_never_shows_tokens():
    record = logging.LogRecord("uvicorn.access", logging.INFO, "", 0, '%s - "%s %s"', (
        "127.0.0.1", "GET", "/api/jobs/1/samples/0/frame.jpg?token=abc.def&x=1"), None)
    HideTokens().filter(record)
    assert record.args[2] == "/api/jobs/1/samples/0/frame.jpg?token=***&x=1"
