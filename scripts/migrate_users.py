# scripts/migrate_users.py
import json
import os
import secrets
from pathlib import Path

from app.auth import create_user, ensure_db, get_user

ROOT = Path(__file__).resolve().parents[1]
USERS_JSON = ROOT / "data" / "users.json"
REPORT = ROOT / "data" / "migration_report.txt"


def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print("Could not read JSON:", e)
        return None


def normalize_users(obj):
    if isinstance(obj, dict):
        if "users" in obj and isinstance(obj["users"], list):
            return obj["users"]
        vals = []
        for v in obj.values():
            if isinstance(v, dict) and ("username" in v or "password" in v):
                vals.append(v)
        if vals:
            return vals
        arr = []
        for k, v in obj.items():
            if isinstance(v, str):
                arr.append({"username": k, "password": v})
        if arr:
            return arr
    elif isinstance(obj, list):
        return obj
    return []


def main():
    ensure_db()
    if not USERS_JSON.exists():
        print("No users.json found at", USERS_JSON)
        return
    raw = load_json(USERS_JSON)
    users = normalize_users(raw)
    report_lines = []
    for u in users:
        username = u.get("username") or u.get("user") or u.get("name")
        if not username:
            continue
        password_plain = None
        if "password" in u and isinstance(u["password"], str):
            password_plain = u["password"]
        if not password_plain:
            password_plain = secrets.token_urlsafe(8)
        role = u.get("role", "admin" if username.lower() == "admin" else "user")
        if get_user(username):
            report_lines.append(f"{username}: already exists -> skipped")
            print(username, "exists, skipped")
            continue
        force_reset = not bool("password" in u and isinstance(u["password"], str))
        create_user(username, password_plain, role=role, force_reset=force_reset)
        report_lines.append(
            f"{username}: created, temporary_password={password_plain}, force_reset={force_reset}"
        )
        print("Created", username)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print("Migration done. See", REPORT)


if __name__ == "__main__":
    main()
