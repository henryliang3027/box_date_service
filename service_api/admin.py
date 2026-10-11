"""
admin.py — 用量 dashboard（伺服器端內部使用）
==============================================
提供 /admin 網頁，查詢客戶的成功請求次數、每日趨勢、逐筆紀錄與上傳原圖。

登入：HTTP Basic（瀏覽器會跳出帳密視窗），帳密來自 .env 的 ADMIN_USER / ADMIN_PASSWORD。
ADMIN_PASSWORD 未設定時，所有 /admin 路由回 503（dashboard 停用）。

路由（皆不顯示在 Swagger 文件中）：
  GET /admin                          → dashboard 網頁
  GET /admin/api/customers            → 客戶代號清單（下拉選單用）
  GET /admin/api/summary              → 每個客戶的統計 + 每日趨勢
  GET /admin/api/requests             → 逐筆紀錄（分頁）
  GET /admin/api/requests.csv         → 逐筆紀錄匯出 CSV
  GET /admin/api/images/{record_id}   → 該筆紀錄的上傳原圖
"""

import csv                                     # 匯出 CSV
import io                                      # CSV 寫入記憶體緩衝區
import secrets                                 # 帳密比對（compare_digest 防 timing attack）
from pathlib import Path                       # dashboard HTML 路徑

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from service_api import config
from service_api.services import usage
from service_api.services.auth import API_KEYS   # 已設定的客戶（含還沒有任何請求的客戶）

_DASHBOARD_HTML = Path(__file__).resolve().parent / "static" / "admin_dashboard.html"
_basic = HTTPBasic(realm="Box Date Service Admin")


def require_admin(credentials: HTTPBasicCredentials = Depends(_basic)) -> None:
    """驗證 dashboard 帳密；未設定 ADMIN_PASSWORD 時停用 dashboard。"""
    if not config.ADMIN_PASSWORD:
        raise HTTPException(status_code=503, detail="Dashboard 未啟用：請在 .env 設定 ADMIN_PASSWORD")
    user_ok = secrets.compare_digest(credentials.username.encode(), config.ADMIN_USER.encode())
    pass_ok = secrets.compare_digest(credentials.password.encode(), config.ADMIN_PASSWORD.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="帳號或密碼錯誤",
            headers={"WWW-Authenticate": 'Basic realm="Box Date Service Admin"'},
        )


router = APIRouter(prefix="/admin", dependencies=[Depends(require_admin)], include_in_schema=False)


def _open_db():
    """開啟唯讀連線；資料庫還不存在時回傳 None（dashboard 顯示 0 筆）。"""
    return usage.connect_readonly()


def _bad_time(e: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(e))


@router.get("")
def dashboard_page() -> FileResponse:
    """dashboard 網頁（單一 HTML，不依賴外部 CDN）。"""
    return FileResponse(_DASHBOARD_HTML, media_type="text/html; charset=utf-8")


@router.get("/api/customers")
def list_customers() -> list[str]:
    """api_keys.json 裡的客戶 + 資料庫中出現過的客戶（已移除 key 的舊客戶也查得到）。"""
    customers = set(API_KEYS.values())
    conn = _open_db()
    if conn is not None:
        with conn:
            customers.update(r[0] for r in conn.execute("SELECT DISTINCT customer FROM usage_log"))
        conn.close()
    return sorted(customers)


@router.get("/api/summary")
def summary(
    start:    str | None = Query(None, description="YYYY-MM-DD（含）"),
    end:      str | None = Query(None, description="YYYY-MM-DD（含）"),
    customer: str | None = Query(None),
) -> dict:
    conn = _open_db()
    if conn is None:
        return {"by_customer": [], "daily": []}
    try:
        return {
            "by_customer": usage.query_by_customer(conn, start, end, customer),
            "daily":       usage.query_daily(conn, start, end, customer),
        }
    except ValueError as e:
        raise _bad_time(e)
    finally:
        conn.close()


@router.get("/api/requests")
def requests_page(
    start:    str | None = Query(None),
    end:      str | None = Query(None),
    customer: str | None = Query(None),
    limit:    int = Query(50, ge=1, le=500),
    offset:   int = Query(0, ge=0),
) -> dict:
    conn = _open_db()
    if conn is None:
        return {"total": 0, "rows": []}
    try:
        total, rows = usage.query_requests(conn, start, end, customer, limit=limit, offset=offset)
    except ValueError as e:
        raise _bad_time(e)
    finally:
        conn.close()
    return {"total": total, "rows": rows}


@router.get("/api/requests.csv")
def requests_csv(
    start:    str | None = Query(None),
    end:      str | None = Query(None),
    customer: str | None = Query(None),
) -> Response:
    conn = _open_db()
    rows: list[dict] = []
    if conn is not None:
        try:
            _, rows = usage.query_requests(conn, start, end, customer)
        except ValueError as e:
            raise _bad_time(e)
        finally:
            conn.close()

    fields = ["received_at", "customer", "endpoint", "original_filename", "num_boxes", "image_bytes", "image_path"]
    buf = io.StringIO()
    buf.write("﻿")  # BOM：Excel 開啟中文不亂碼
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)

    filename = f"usage_{start or 'all'}_{end or 'now'}{'_' + customer if customer else ''}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/api/images/{record_id}")
def record_image(record_id: int) -> FileResponse:
    conn = _open_db()
    if conn is None:
        raise HTTPException(status_code=404, detail="找不到圖片")
    try:
        path = usage.get_image_path(conn, record_id)
    finally:
        conn.close()
    if path is None:
        raise HTTPException(status_code=404, detail="找不到圖片")
    return FileResponse(path)
