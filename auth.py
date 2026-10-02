# auth.py
# Handles users, password hashing, login tokens, and the role hierarchy.

import sqlite3
import os
import secrets
from datetime import datetime, timedelta, timezone
from passlib.context import CryptContext
from jose import JWTError, jwt

# ── SECRET KEY ────────────────────────────────────────────────────────────────
# This key signs every JWT token. If an attacker gets this key they can forge
# admin tokens and bypass all access control.
#
# HOW TO SET IT SAFELY:
#   Option A (recommended) — set an environment variable before starting:
#     Windows:  set JWT_SECRET_KEY=some_long_random_string
#     Linux:    export JWT_SECRET_KEY=some_long_random_string
#   Option B — generate a strong key once and paste it below:
#     Run in Python:  import secrets; print(secrets.token_hex(32))
#
# NEVER use the placeholder string in production.
SECRET_KEY = os.environ.get("JWT_SECRET_KEY") or secrets.token_hex(32)
# ↑ If the environment variable is not set, a new random key is generated
#   each time the server starts. Tokens from previous sessions will be
#   invalid after a restart, which is safe — users just log in again.

ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 8

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# ── ROLE HIERARCHY ────────────────────────────────────────────────────────────
# This is the single source of truth for all roles in the system.
#
# HOW TO ADD A NEW ROLE:
#   1. Add it here with the next number: "ceo": 4, "manager": 2, etc.
#   2. Add a test account in TEST_USERS below (optional).
#   3. That's it — permissions, upload rights, and the UI all update automatically.
#
# The number is the "clearance level". A role can read documents tagged with
# any level that is less than or equal to its own level.
# Example: employee (level 1) can read "guest" (0) and "employee" (1) docs.

ROLE_HIERARCHY: dict[str, int] = {
    "public":    0,
    "employee": 1,
    "admin":    2,
    # "manager": 3,   ← uncomment and renumber to add a new role
    # "ceo":     4,
}

# Roles allowed to upload documents.
# Change this list to control who can upload — no other code needs to change.
UPLOAD_ROLES: set[str] = {"admin"}
# Example with more roles:  {"admin", "ceo", "manager"}


def can_upload(role: str) -> bool:
    """Returns True if this role is allowed to upload documents."""
    return role in UPLOAD_ROLES


def get_allowed_levels(role: str) -> list[str]:
    """
    Returns the list of access_level tags this role is allowed to read.
    A role can read its own level and all levels below it.
    Example: employee (level 1) → ["guest", "employee"]
    """
    user_level = ROLE_HIERARCHY.get(role, -1)
    return [r for r, level in ROLE_HIERARCHY.items() if level <= user_level]


def get_all_access_levels() -> list[str]:
    """
    Returns all access level names sorted from lowest to highest.
    Used by the upload UI to populate the dropdown.
    """
    return sorted(ROLE_HIERARCHY.keys(), key=lambda r: ROLE_HIERARCHY[r])


# ── DATABASE ──────────────────────────────────────────────────────────────────

# Test accounts created automatically on first run.
# Remove or change these before deploying to real users.
TEST_USERS = [
    ("admin_user",    "admin123",    "admin"),
    ("employee_user", "employee123", "employee"),
    ("guest_user",    "guest123",    "public"),
    ("Victoria",    "112358",    "admin"),
]


def init_database():
    conn   = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role          TEXT NOT NULL DEFAULT 'guest'
        )
    """)

    # Login attempt tracking — used for brute-force protection
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_attempts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT NOT NULL,
            ip_address TEXT,
            success    INTEGER NOT NULL DEFAULT 0,
            attempted_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    for username, password, role in TEST_USERS:
        if not cursor.execute(
            "SELECT id FROM users WHERE username = ?", (username,)
        ).fetchone():
            cursor.execute(
                "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                (username, pwd_context.hash(password), role)
            )
            print(f"✓ Created test user: {username} / {password} [{role}]")

    conn.commit()
    conn.close()


# ── BRUTE-FORCE PROTECTION ────────────────────────────────────────────────────
# Tracks failed login attempts per username in the database.
# After MAX_FAILURES failures in WINDOW_MINUTES minutes, the account is
# temporarily locked. This makes automated password-guessing impractical.

MAX_FAILURES     = 5    # failed attempts before lockout
WINDOW_MINUTES   = 15   # sliding time window
LOCKOUT_MINUTES  = 15   # how long the lockout lasts


def record_login_attempt(username: str, success: bool, ip: str = None):
    conn = sqlite3.connect("users.db")
    conn.execute(
        "INSERT INTO login_attempts (username, ip_address, success) VALUES (?, ?, ?)",
        (username, ip, 1 if success else 0)
    )
    conn.commit()
    conn.close()


def is_account_locked(username: str) -> bool:
    """
    Returns True if this username has too many recent failed logins.
    Only counts failures in the last WINDOW_MINUTES minutes.
    """
    conn   = sqlite3.connect("users.db")
    cursor = conn.cursor()

    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=WINDOW_MINUTES)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    failures = cursor.execute(
        """SELECT COUNT(*) FROM login_attempts
           WHERE username = ? AND success = 0 AND attempted_at >= ?""",
        (username, cutoff)
    ).fetchone()[0]

    conn.close()
    return failures >= MAX_FAILURES


# ── USER OPERATIONS ───────────────────────────────────────────────────────────

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def get_user(username: str):
    conn = sqlite3.connect("users.db")
    row  = conn.execute(
        "SELECT username, password_hash, role FROM users WHERE username = ?",
        (username,)
    ).fetchone()
    conn.close()
    if row:
        return {"username": row[0], "password_hash": row[1], "role": row[2]}
    return None


def authenticate_user(username: str, password: str, ip: str = None):
    """
    Validates credentials. Returns user dict on success, False on failure.
    Records every attempt for brute-force tracking.
    Refuses immediately if the account is currently locked.
    """
    if is_account_locked(username):
        return "locked"   # distinct from False so the caller can give a better message

    user = get_user(username)
    if not user or not verify_password(password, user["password_hash"]):
        record_login_attempt(username, success=False, ip=ip)
        return False

    record_login_attempt(username, success=True, ip=ip)
    return user


# ── TOKEN OPERATIONS ──────────────────────────────────────────────────────────

# Active token store — maps username → set of valid token IDs.
# When a user logs out, their token ID is removed from this set.
# Any request carrying that token ID will then be rejected,
# even if the token hasn't expired yet.
_active_tokens: dict[str, set[str]] = {}


def create_token(username: str, role: str) -> str:
    token_id = secrets.token_hex(16)   # unique ID for this specific token
    expire   = datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRE_HOURS)

    payload = {
        "sub":  username,
        "role": role,
        "exp":  expire,
        "jti":  token_id,   # "jti" = JWT ID, standard field
    }
    token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    # Register this token as active
    _active_tokens.setdefault(username, set()).add(token_id)
    return token


def revoke_token(token: str):
    """
    Invalidates a specific token immediately — used on logout.
    Even if the token hasn't expired yet, it will be rejected.
    """
    payload = _decode_raw(token)
    if payload:
        username = payload.get("sub")
        jti      = payload.get("jti")
        if username and jti and username in _active_tokens:
            _active_tokens[username].discard(jti)


def decode_token(token: str):
    """
    Validates a token. Returns payload only if:
    - Signature is valid
    - Token has not expired
    - Token has not been revoked (still in _active_tokens)
    """
    payload = _decode_raw(token)
    if not payload:
        return None

    username = payload.get("sub")
    jti      = payload.get("jti")

    # Reject if token was revoked (e.g. user logged out)
    if not username or not jti:
        return None
    if jti not in _active_tokens.get(username, set()):
        return None

    return payload


def _decode_raw(token: str):
    """Decodes and verifies signature/expiry without checking revocation."""
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        return None


def revoke_all_tokens(username: str):
    """Invalidates all tokens for a user — e.g. on password change."""
    _active_tokens.pop(username, None)