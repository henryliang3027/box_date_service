"""
services/web_auth.py — 手機 Web 版（/app）帳號密碼登入與 session
==================================================================
Web 版給客戶的「現場人員」用：一人一組帳號，每個帳號對應一個客戶代號（同 api_keys.json 的 value），
用量照樣記在該客戶名下。API Key 不會傳到瀏覽器。

  - 帳號：web_users.json（{ 帳號: {"customer": 客戶代號, "password": 雜湊} }），
    用專案根目錄的 manage_web_users.py 新增 / 改密碼 / 停用。檔案修改後不用重啟。
  - 密碼：PBKDF2-SHA256 + 隨機 salt，只存雜湊。
  - session：登入成功產生隨機 token 放進 HttpOnly cookie；資料庫只存 token 的 SHA-256，
    資料庫外流也無法拿來登入。每次請求都會重新確認帳號仍存在、客戶代號沒變，
    所以從 web_users.json 移除帳號後，該帳號已登入的手機立即失效。
  - 暴力破解：同一帳號連續失敗 5 次，鎖 15 分鐘（記在記憶體，重啟歸零）。
"""

import hashlib                                  # 密碼 / token 雜湊
import hmac                                     # 密碼比對（compare_digest）
import json                                     # 讀取 web_users.json
import os                                       # 檔案修改時間（帳號清單快取）
import secrets                                  # session token / salt
import sqlite3                                  # session 資料庫
import threading                                # 失敗次數計數的鎖
import time                                     # 鎖定時間計算
from datetime import datetime, timedelta        # session 到期時間
from pathlib import Path                        # 資料庫路徑

from fastapi import Cookie, HTTPException       # 從 cookie 取 token、401 回應

from service_api import config                  # WEB_USERS_PATH、WEB_SESSIONS_DB_PATH、WEB_SESSION_HOURS

SESSION_COOKIE = "bds_session"                  # cookie 名稱

_PBKDF2_ITERATIONS = 600_000
_MAX_FAILURES      = 5                          # 連續失敗幾次後鎖定
_LOCK_SECONDS      = 15 * 60                    # 鎖定多久

_SCHEMA = """
CREATE TABLE IF NOT EXISTS web_sessions (
    token_hash  TEXT PRIMARY KEY,   -- session token 的 SHA-256（不存明文 token）
    username    TEXT NOT NULL,
    customer    TEXT NOT NULL,
    created_at  TEXT NOT NULL,      -- YYYY-MM-DD HH:MM:SS
    expires_at  TEXT NOT NULL       -- YYYY-MM-DD HH:MM:SS
);
CREATE INDEX IF NOT EXISTS idx_web_sessions_user ON web_sessions(username);
"""


# ── 密碼雜湊 ──────────────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """回傳 "pbkdf2_sha256$<iterations>$<salt hex>$<hash hex>"。"""
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, _PBKDF2_ITERATIONS)
    return f"pbkdf2_sha256${_PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_hex, hash_hex = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations))
    except (ValueError, AttributeError):
        return False
    return hmac.compare_digest(digest.hex(), hash_hex)


# 帳號不存在時也跑一次雜湊，讓「帳號不存在」和「密碼錯誤」的回應時間一樣，無法用時間猜帳號
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


# ── 帳號清單（web_users.json，依修改時間快取）──────────────────────────────────

_users_cache: tuple[float, dict] | None = None


def load_users() -> dict[str, dict]:
    """讀取帳號清單；檔案不存在或格式錯誤時回傳空 dict（全部無法登入，fail closed）。"""
    global _users_cache
    try:
        mtime = os.path.getmtime(config.WEB_USERS_PATH)
    except OSError:
        return {}
    if _users_cache and _users_cache[0] == mtime:
        return _users_cache[1]
    try:
        with open(config.WEB_USERS_PATH, encoding="utf-8") as f:
            users = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"⚠️ [web_auth] 無法讀取 {config.WEB_USERS_PATH}（{e}），Web 版無法登入")
        return {}
    _users_cache = (mtime, users)
    return users


# ── 登入失敗鎖定 ──────────────────────────────────────────────────────────────

_failures: dict[str, tuple[int, float]] = {}   # 帳號 → (連續失敗次數, 最後一次失敗時間)
_failures_lock = threading.Lock()


def _locked_out(username: str) -> bool:
    with _failures_lock:
        count, last = _failures.get(username, (0, 0.0))
        if count >= _MAX_FAILURES and time.time() - last < _LOCK_SECONDS:
            return True
        if count >= _MAX_FAILURES:
            _failures.pop(username, None)       # 鎖定時間已過，重新計算
        return False


def _record_failure(username: str) -> None:
    with _failures_lock:
        count, _ = _failures.get(username, (0, 0.0))
        _failures[username] = (count + 1, time.time())


def authenticate(username: str, password: str) -> str:
    """帳密正確回傳客戶代號；錯誤回 401，鎖定中回 429。"""
    if _locked_out(username):
        raise HTTPException(status_code=429, detail="登入失敗次數過多，請 15 分鐘後再試")
    user = load_users().get(username)
    ok = verify_password(password, user["password"] if user else _DUMMY_HASH)
    if not (user and ok):
        _record_failure(username)
        raise HTTPException(status_code=401, detail="帳號或密碼錯誤")
    with _failures_lock:
        _failures.pop(username, None)
    return user["customer"]


# ── session ───────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    Path(config.WEB_SESSIONS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.WEB_SESSIONS_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def create_session(username: str, customer: str) -> str:
    """建立 session，回傳要放進 cookie 的 token；順便清掉已過期的 session。"""
    token = secrets.token_urlsafe(32)
    expires = (datetime.now() + timedelta(hours=config.WEB_SESSION_HOURS)).strftime("%Y-%m-%d %H:%M:%S")
    with _connect() as conn:
        conn.execute("DELETE FROM web_sessions WHERE expires_at <= ?", (_now(),))
        conn.execute(
            "INSERT INTO web_sessions (token_hash, username, customer, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (_token_hash(token), username, customer, _now(), expires),
        )
    conn.close()
    return token


def delete_session(token: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM web_sessions WHERE token_hash = ?", (_token_hash(token),))
    conn.close()


def delete_user_sessions(username: str) -> int:
    """讓某帳號所有已登入的裝置登出（改密碼 / 停用帳號時用），回傳刪除筆數。"""
    with _connect() as conn:
        n = conn.execute("DELETE FROM web_sessions WHERE username = ?", (username,)).rowcount
    conn.close()
    return n


def require_web_user(bds_session: str | None = Cookie(None)) -> tuple[str, str]:
    """
    路由依賴：驗證 session cookie，通過時回傳 (帳號, 客戶代號)，否則 401。
    帳號已從 web_users.json 移除、或客戶代號被改掉時，舊 session 一律無效。
    """
    if not bds_session:
        raise HTTPException(status_code=401, detail="請先登入")
    with _connect() as conn:
        row = conn.execute(
            "SELECT username, customer FROM web_sessions WHERE token_hash = ? AND expires_at > ?",
            (_token_hash(bds_session), _now()),
        ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=401, detail="登入已過期，請重新登入")
    username, customer = row
    user = load_users().get(username)
    if user is None or user.get("customer") != customer:
        raise HTTPException(status_code=401, detail="帳號已停用，請聯絡管理員")
    return username, customer
