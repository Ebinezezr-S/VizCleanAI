import sqlite3

import bcrypt

c = sqlite3.connect("data/users.db")
row = c.execute(
    "SELECT username, password_hash FROM users WHERE username='admin'"
).fetchone()
c.close()
print("DB row:", row)
if row:
    stored = row[1].encode()
    print("bcrypt check:", bcrypt.checkpw(b"Admin@123", stored))
else:
    print("No admin user found")
