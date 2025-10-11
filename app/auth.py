import datetime
import os
import sqlite3
from types import SimpleNamespace

import bcrypt

DB_PATH = os.path.join("data", "users.db")


def ensure_db():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # create users table if missing, including force_reset
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            force_reset INTEGER DEFAULT 0
        )
    """
    )
    # try adding column if older table exists without it (safe)
    try:
        c.execute("ALTER TABLE users ADD COLUMN force_reset INTEGER DEFAULT 0")
    except Exception:
        # ignore if column already exists or cannot be added
        pass
    conn.commit()
    conn.close()


def get_user(username):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # make sure schema up-to-date
    try:
        c.execute("ALTER TABLE users ADD COLUMN force_reset INTEGER DEFAULT 0")
        conn.commit()
    except Exception:
        pass
    c.execute(
        "SELECT id, username, password_hash, created_at, force_reset FROM users WHERE LOWER(username)=LOWER(?)",
        (username,),
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    # return a SimpleNamespace instance so attributes exist reliably
    uid, uname, pw_hash, created_at, force_reset = row
    return SimpleNamespace(
        id=uid,
        username=uname,
        password_hash=pw_hash,
        created_at=created_at,
        force_reset=bool(force_reset),
    )


def create_user(username, password, force_reset=False):
    ensure_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    c.execute(
        "INSERT OR REPLACE INTO users (username, password_hash, created_at, force_reset) VALUES (?, ?, ?, ?)",
        (
            username.lower(),
            hashed,
            datetime.datetime.now().isoformat(),
            1 if force_reset else 0,
        ),
    )
    conn.commit()
    conn.close()


def verify_user(username, password):
    username = username.lower().strip()
    user = get_user(username)
    if not user:
        return False
    stored_hash = user.password_hash.encode()
    return bcrypt.checkpw(password.encode(), stored_hash)


def change_password(username, new_password):
    username = username.lower().strip()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    new_hash = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
    c.execute(
        "UPDATE users SET password_hash=?, force_reset=0 WHERE LOWER(username)=LOWER(?)",
        (new_hash, username),
    )
    conn.commit()
    conn.close()
