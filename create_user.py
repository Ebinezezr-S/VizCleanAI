# create_user.py
import getpass
import hashlib
import json
import os
import secrets

ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, "data")
USERS_PATH = os.path.join(DATA_DIR, "users.json")

os.makedirs(DATA_DIR, exist_ok=True)


def hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


def load_users():
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, "r", encoding="utf8") as f:
            return json.load(f)
    return {}


def save_users(users: dict):
    with open(USERS_PATH, "w", encoding="utf8") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)


def create_user(username: str, password: str):
    users = load_users()
    if username in users:
        raise ValueError("User already exists")
    salt = secrets.token_hex(8)
    pw_hash = hash_pw(password, salt)
    users[username] = {"salt": salt, "pw_hash": pw_hash}
    save_users(users)
    print(f"Created user: {username}")


if __name__ == "__main__":
    print("Create new user for VizClean local auth")
    uname = input("Username: ").strip()
    pw = getpass.getpass("Password: ").strip()
    pw2 = getpass.getpass("Confirm password: ").strip()
    if pw != pw2:
        print("Passwords do not match. Aborting.")
    else:
        create_user(uname, pw)
