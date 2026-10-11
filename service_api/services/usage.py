"""
services/usage.py — 客戶用量紀錄（計費依據）
==============================================
每次「成功」的請求記一筆到 SQLite，並把客戶上傳的原圖存檔，供日後查詢：
  - 哪個客戶、什麼時間、傳了哪張圖（原始檔名 + 伺服器存檔路徑）
  - 指定時間區間內每個客戶的請求次數

「成功」的定義：
  record_usage() 由路由以 BackgroundTasks 排程，Starlette 只會在 response
  完整送出「之後」才執行背景任務；路由拋出 HTTPException（401/400）或
  發生例外（500）時，背景任務不會執行，也就不會記錄。
  因此只有 200 且 response 已送達的請求才會被計數。

查詢介面：專案根目錄的 usage_report.py（命令列）、/admin（dashboard 網頁）
"""

import sqlite3                    # 用量資料庫
from datetime import datetime, timedelta  # 存檔路徑 / 紀錄時間 / 查詢區間
from pathlib import Path          # 圖片存檔路徑操作

from service_api import config    # USAGE_DB_PATH、USAGE_IMAGES_DIR

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_log (
    id                INTEGER PRIMARY KEY,
    customer          TEXT    NOT NULL,   -- 客戶代號（api_keys.json 的 value）
    endpoint          TEXT    NOT NULL,   -- 呼叫的 API 路徑
    received_at       TEXT    NOT NULL,   -- 收到請求的本地時間 YYYY-MM-DD HH:MM:SS.ffffff
    original_filename TEXT,               -- 客戶上傳時的檔名
    image_path        TEXT,               -- 伺服器存檔路徑（相對於 USAGE_IMAGES_DIR）；存檔失敗為 NULL
    image_bytes       INTEGER NOT NULL,   -- 圖片大小（bytes）
    num_boxes         INTEGER NOT NULL    -- 該次辨識回傳的紙箱數
);
CREATE INDEX IF NOT EXISTS idx_usage_customer_time ON usage_log(customer, received_at);
CREATE INDEX IF NOT EXISTS idx_usage_time          ON usage_log(received_at);
"""


def connect() -> sqlite3.Connection:
    """開啟用量資料庫連線（不存在時自動建立資料夾與資料表）。"""
    Path(config.USAGE_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(config.USAGE_DB_PATH, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")  # 多個 worker 同時寫入不互卡
    conn.executescript(_SCHEMA)
    return conn


def record_usage(
    customer:          str,       # 客戶代號
    endpoint:          str,       # API 路徑
    received_at:       datetime,  # 收到請求的時間
    original_filename: str,       # 客戶上傳時的檔名
    image_bytes:       bytes,     # 原圖 bytes
    num_boxes:         int,       # 辨識出的紙箱數
) -> None:
    """
    存原圖到 USAGE_IMAGES_DIR/<客戶>/<YYYYMMDD>/<時間戳><副檔名>，並寫入一筆 usage_log。
    以 BackgroundTasks 執行，不影響回應時間；圖片存檔失敗時仍會寫入紀錄（image_path=NULL），
    避免漏計請求次數。
    """
    image_rel_path: str | None = None
    try:
        suffix   = Path(original_filename).suffix.lower() or ".jpg"
        rel_path = Path(customer) / received_at.strftime("%Y%m%d") / f"{received_at.strftime('%H%M%S_%f')}{suffix}"
        out_path = Path(config.USAGE_IMAGES_DIR) / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(image_bytes)
        image_rel_path = str(rel_path)
    except OSError as e:
        print(f"⚠️ [usage] 圖片存檔失敗（customer={customer}）：{e}")

    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO usage_log "
                "(customer, endpoint, received_at, original_filename, image_path, image_bytes, num_boxes) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    customer,
                    endpoint,
                    received_at.strftime("%Y-%m-%d %H:%M:%S.%f"),
                    original_filename,
                    image_rel_path,
                    len(image_bytes),
                    num_boxes,
                ),
            )
    except sqlite3.Error as e:
        print(f"❌ [usage] 用量紀錄寫入失敗（customer={customer} received_at={received_at}）：{e}")


# ── 查詢（usage_report.py 與 /admin dashboard 共用）─────────────────────────────

def parse_time(value: str, is_end: bool) -> str:
    """
    把 YYYY-MM-DD 或 "YYYY-MM-DD HH:MM:SS" 轉成可直接與 received_at 字串比較的時間字串。
    起訖皆為包含：只給日期時 end 包含當天整天，精確到秒時包含該秒。
    格式錯誤時拋出 ValueError。
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            t = datetime.strptime(value, fmt)
        except ValueError:
            continue
        if is_end:
            t += timedelta(days=1) if fmt == "%Y-%m-%d" else timedelta(seconds=1)
        return t.strftime("%Y-%m-%d %H:%M:%S.%f")
    raise ValueError(f"時間格式錯誤：{value}（請用 YYYY-MM-DD 或 \"YYYY-MM-DD HH:MM:SS\"）")


def _where(start: str | None, end: str | None, customer: str | None) -> tuple[str, list]:
    """組出 WHERE 子句；end 為「不含」上界（已由 parse_time 往後推）。"""
    clauses, params = [], []
    if start:
        clauses.append("received_at >= ?")
        params.append(parse_time(start, is_end=False))
    if end:
        clauses.append("received_at < ?")
        params.append(parse_time(end, is_end=True))
    if customer:
        clauses.append("customer = ?")
        params.append(customer)
    return ("WHERE " + " AND ".join(clauses)) if clauses else "", params


def connect_readonly() -> sqlite3.Connection | None:
    """以唯讀模式開啟用量資料庫；資料庫尚不存在（還沒有任何成功請求）時回傳 None。"""
    if not Path(config.USAGE_DB_PATH).exists():
        return None
    return sqlite3.connect(f"file:{config.USAGE_DB_PATH}?mode=ro", uri=True)


def query_by_customer(conn: sqlite3.Connection, start=None, end=None, customer=None) -> list[dict]:
    """每個客戶的請求次數、總箱數、第一筆 / 最後一筆時間。"""
    where_sql, params = _where(start, end, customer)
    rows = conn.execute(
        f"SELECT customer, COUNT(*), SUM(num_boxes), MIN(received_at), MAX(received_at) "
        f"FROM usage_log {where_sql} GROUP BY customer ORDER BY customer",
        params,
    ).fetchall()
    return [
        {"customer": r[0], "requests": r[1], "total_boxes": r[2], "first_request": r[3][:19], "last_request": r[4][:19]}
        for r in rows
    ]


def query_daily(conn: sqlite3.Connection, start=None, end=None, customer=None) -> list[dict]:
    """
    每日請求次數與箱數，沒有請求的日子補 0（區間：start 或第一筆 ~ end 或今天）。
    """
    where_sql, params = _where(start, end, customer)
    rows = conn.execute(
        f"SELECT substr(received_at, 1, 10) AS day, COUNT(*), SUM(num_boxes) "
        f"FROM usage_log {where_sql} GROUP BY day ORDER BY day",
        params,
    ).fetchall()
    by_day = {r[0]: (r[1], r[2]) for r in rows}
    if not by_day and not start:
        return []

    first = datetime.strptime(start, "%Y-%m-%d").date() if start else datetime.strptime(rows[0][0], "%Y-%m-%d").date()
    last  = datetime.strptime(end[:10], "%Y-%m-%d").date() if end else datetime.now().date()
    days, d = [], first
    while d <= last:
        key = d.isoformat()
        requests, boxes = by_day.get(key, (0, 0))
        days.append({"day": key, "requests": requests, "total_boxes": boxes})
        d += timedelta(days=1)
    return days


def query_requests(
    conn: sqlite3.Connection, start=None, end=None, customer=None, limit: int | None = None, offset: int = 0,
) -> tuple[int, list[dict]]:
    """逐筆請求（新到舊），回傳 (符合條件的總筆數, 這一頁的資料)。"""
    where_sql, params = _where(start, end, customer)
    total = conn.execute(f"SELECT COUNT(*) FROM usage_log {where_sql}", params).fetchone()[0]
    limit_sql = f"LIMIT {int(limit)} OFFSET {int(offset)}" if limit is not None else ""
    rows = conn.execute(
        f"SELECT id, received_at, customer, endpoint, original_filename, num_boxes, image_bytes, image_path "
        f"FROM usage_log {where_sql} ORDER BY received_at DESC {limit_sql}",
        params,
    ).fetchall()
    return total, [
        {
            "id": r[0], "received_at": r[1][:19], "customer": r[2], "endpoint": r[3],
            "original_filename": r[4], "num_boxes": r[5], "image_bytes": r[6], "image_path": r[7],
        }
        for r in rows
    ]


def get_image_path(conn: sqlite3.Connection, record_id: int) -> Path | None:
    """依紀錄 id 取得原圖的絕對路徑；找不到紀錄、沒有存檔、或路徑跑出 USAGE_IMAGES_DIR 時回傳 None。"""
    row = conn.execute("SELECT image_path FROM usage_log WHERE id = ?", (record_id,)).fetchone()
    if not row or not row[0]:
        return None
    base = Path(config.USAGE_IMAGES_DIR).resolve()
    path = (base / row[0]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        return None
    return path
