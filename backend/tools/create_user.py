"""Add or update a dashboard user in config/users.json (PBKDF2 password hashes only; the file is git-ignored).

    cd backend
    ..\\venv\\Scripts\\python.exe -m tools.create_user admin1 --role admin --name "Control Room Admin"   # asks for a password
    ..\\venv\\Scripts\\python.exe -m tools.create_user officer1 --role officer --generate   # prints a new password once
    ..\\venv\\Scripts\\python.exe -m tools.create_user --list
    ..\\venv\\Scripts\\python.exe -m tools.create_user officer1 --delete

Roles: admin (Control Room Admin: everything, incl. video upload, test cameras, reports) and officer
(Security Officer: command centre, acknowledging alerts). An existing username gets the new password,
name and role. If .env has no SECRET_KEY (it signs the session tokens), one is generated and added.
The backend reads the users file again when it changes: no restart needed.
"""

from __future__ import annotations

import argparse
import getpass
import json
import secrets
import string
from pathlib import Path

from app.auth import ROLES, hash_password
from app.config import REPO_ROOT, get_settings

MIN_LENGTH = 10


def ensure_secret_key(env_file: Path = REPO_ROOT / ".env") -> bool:
    """Append a random SECRET_KEY to .env if it has none; True if one was added."""
    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.exists() else []
    if any(line.split("=", 1)[0].strip() == "SECRET_KEY" and line.split("=", 1)[1].strip()
           for line in lines if "=" in line and not line.lstrip().startswith("#")):
        return False
    text = env_file.read_text(encoding="utf-8") if env_file.exists() else ""
    if text and not text.endswith("\n"):
        text += "\n"
    text += ("\n# Signs the dashboard's login sessions (added by tools/create_user.py). Keep it secret; changing it\n"
             f"# logs everyone out.\nSECRET_KEY={secrets.token_urlsafe(48)}\n")
    env_file.write_text(text, encoding="utf-8")
    return True


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"users": []}


def save(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def generate_password(length: int = 14) -> str:
    alphabet = string.ascii_letters + string.digits
    while True:
        word = "".join(secrets.choice(alphabet) for _ in range(length))
        if any(c.isdigit() for c in word) and any(c.isupper() for c in word) and any(c.islower() for c in word):
            return word


def upsert(data: dict, username: str, name: str, role: str, password: str) -> None:
    users = [u for u in data["users"] if u["username"].lower() != username.lower()]
    users.append({"username": username, "name": name, "role": role, "password_hash": hash_password(password)})
    data["users"] = sorted(users, key=lambda u: u["username"].lower())


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("username", nargs="?")
    p.add_argument("--role", choices=sorted(ROLES), default="officer")
    p.add_argument("--name", help="name shown in the dashboard (default: the username)")
    p.add_argument("--password", help="avoid: it stays in your shell history; omit to be asked")
    p.add_argument("--generate", action="store_true", help="generate a strong password and print it once")
    p.add_argument("--list", action="store_true", help="list the users")
    p.add_argument("--delete", action="store_true", help="remove the user")
    args = p.parse_args(argv)
    path = get_settings().users_config
    if ensure_secret_key():
        print("Added a new SECRET_KEY to .env (restart the backend once to use it).")
    data = load(path)

    if args.list or not args.username:
        for u in data["users"]:
            print(f"{u['username']:<16} {u['role']:<8} {u.get('name', '')}")
        print(f"{len(data['users'])} user(s) in {path}")
        return
    if args.delete:
        before = len(data["users"])
        data["users"] = [u for u in data["users"] if u["username"].lower() != args.username.lower()]
        save(path, data)
        print(f"removed {args.username}" if len(data["users"]) < before else f"no user {args.username}")
        return

    if args.generate:
        password = generate_password()
    elif args.password:
        password = args.password
    else:
        password = getpass.getpass(f"Password for {args.username}: ")
        if password != getpass.getpass("Again: "):
            raise SystemExit("The passwords do not match.")
    if len(password) < MIN_LENGTH:
        raise SystemExit(f"Use at least {MIN_LENGTH} characters.")
    upsert(data, args.username, args.name or args.username, args.role, password)
    save(path, data)
    print(f"saved {args.username} ({ROLES[args.role]}) in {path}")
    if args.generate:
        print(f"password (shown once, not stored): {password}")


if __name__ == "__main__":
    main()
