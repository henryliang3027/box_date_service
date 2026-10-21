"""
webapp.py — 手機 Web 版（給客戶現場人員用）
=============================================
現場人員用手機瀏覽器開 /app，帳號密碼登入後拍照，網頁把照片傳到 /app/api/detect，
伺服器跑同一套辨識流程，結果（標注圖、品項數量、每箱日期）直接顯示在網頁上。

  - 登入：帳號密碼 → HttpOnly session cookie（見 services/web_auth.py），不使用也不外流 API Key
  - 用量：與 /api/v1/detect/image 一樣只記成功的請求，endpoint 欄位為 /app/api/detect，
    並多記是哪個帳號拍的（usage_log.username）

路由（皆不顯示在 Swagger 文件中）：
  GET  /app               → Web 版網頁（未登入時網頁自己顯示登入畫面）
  POST /app/api/login     → 帳號密碼登入，設定 session cookie
  POST /app/api/logout    → 登出
  GET  /app/api/me        → 目前登入的帳號 / 客戶
  POST /app/api/detect    → 上傳照片辨識（需登入）
"""

import time                                     # 辨識完成時間戳（給 TimingMiddleware）
from datetime import datetime                   # 用量紀錄的收到時間
from pathlib import Path                        # 網頁 HTML 路徑
from typing import Annotated

import cv2                                      # 驗證上傳圖片可解碼
import numpy as np

from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

from service_api import config
from service_api.schemas import DetectResponse
from service_api.services import web_auth
from service_api.services.usage import record_usage

_APP_HTML = Path(__file__).resolve().parent / "static" / "web_app.html"

router = APIRouter(prefix="/app", include_in_schema=False)


class LoginRequest(BaseModel):
    username: str
    password: str


@router.get("")
def app_page() -> FileResponse:
    # no-cache：更新網頁後手機不會一直拿到舊版
    return FileResponse(_APP_HTML, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-cache"})


@router.post("/api/login")
def login(body: LoginRequest, response: Response) -> dict:
    username = body.username.strip()
    customer = web_auth.authenticate(username, body.password)
    token = web_auth.create_session(username, customer)
    response.set_cookie(
        web_auth.SESSION_COOKIE,
        token,
        max_age  = config.WEB_SESSION_HOURS * 3600,
        path     = "/app",
        httponly = True,                        # 網頁 JavaScript 讀不到
        secure   = config.WEB_COOKIE_SECURE,    # 只在 HTTPS 送出
        samesite = "lax",                       # 其他網站無法帶著這個 cookie 發 POST
    )
    return {"username": username, "customer": customer}


@router.post("/api/logout")
def logout(response: Response, bds_session: str | None = Cookie(None)) -> dict:
    if bds_session:
        web_auth.delete_session(bds_session)
    response.delete_cookie(web_auth.SESSION_COOKIE, path="/app")
    return {"ok": True}


@router.get("/api/me")
def me(user: tuple[str, str] = Depends(web_auth.require_web_user)) -> dict:
    username, customer = user
    return {"username": username, "customer": customer}


@router.post("/api/detect", response_model=DetectResponse)
def detect(
    request: Request,
    file: Annotated[UploadFile, File()],
    background_tasks: BackgroundTasks,
    user: tuple[str, str] = Depends(web_auth.require_web_user),
) -> DetectResponse:
    username, customer = user
    if config.DEBUG:
        print(f"[DEBUG] /app/api/detect 圖片接收時間：{datetime.now().strftime('%H:%M:%S')}  customer={customer} user={username}")

    image_bytes = file.file.read()
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="無效的圖片格式。")

    result = request.app.state.pipeline.run(
        image_bytes      = image_bytes,
        filename         = file.filename or "camera.jpg",
        include_image    = True,                # 網頁一定要顯示標注圖
        print_confidence = True,
    )
    request.state.t_inference_done = time.time()

    # 與 /api/v1/detect/image 相同：response 成功送出後才記錄，失敗的請求不計
    background_tasks.add_task(
        record_usage,
        customer          = customer,
        endpoint          = request.url.path,
        received_at       = datetime.fromtimestamp(request.state.t_recv),
        original_filename = file.filename or "camera.jpg",
        image_bytes       = image_bytes,
        num_boxes         = result.total_boxes,
        username          = username,
    )
    return result
