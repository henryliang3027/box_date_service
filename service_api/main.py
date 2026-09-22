"""
main.py — FastAPI 應用程式入口
================================
定義 API 路由、啟動/關閉生命週期（模型載入）。

設計原則：
  - 路由層（此檔案）只負責 HTTP 的部分：接收請求、驗證輸入、回傳回應
  - 所有業務邏輯都在 pipeline.py；不在此檔案寫 YOLO / Gemini 相關程式碼
  - 模型載入在 lifespan 的 startup 階段完成，不在路由函式內重複載入

啟動指令（在專案根目錄執行）：
  uvicorn service_api.main:app --host 0.0.0.0 --port 8080
  uvicorn service_api.main:app --host 0.0.0.0 --port 8080 --reload   # 開發模式

API 端點一覽：
  GET  /health                        → 健康檢查
  POST /api/v1/detect                 → 偵測（回傳 JSON）
  POST /api/v1/detect/image           → 偵測（回傳 JSON，含標注圖 base64 + mask 座標）
  POST /api/v1/detect/show_image      → 偵測（直接回傳標注後的 JPEG 圖片 binary）

文件：
  啟動後開啟 http://localhost:8080/docs 可看 Swagger UI
"""

from contextlib import asynccontextmanager   # 用於定義 lifespan（startup/shutdown 鉤子）
from pathlib import Path                     # 儲存 mask 圖片的路徑操作
from typing import Annotated                 # 用於 FastAPI 的依賴注入型別標註

import base64                                # 解碼 base64 圖片（/detect/image 端點用）
import time                                  # latency 測量用的高精度時間戳（epoch seconds）
import uuid                                  # 產生不重複的 mask 檔名
from datetime import datetime                # DEBUG 模式下印出圖片接收時間

import cv2                                   # 驗證上傳圖片是否可解碼
import numpy as np                           # 圖片 bytes 轉 numpy array

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Request, UploadFile  # FastAPI 核心元件
from fastapi.responses import Response       # 直接回傳二進位圖片
from starlette.middleware.base import BaseHTTPMiddleware  # latency 測量中介層

from service_api import config               # 所有設定常數
from service_api.pipeline import BoxDetectionPipeline  # 核心業務流程
from service_api.schemas import (
    DetectRequest,     # POST /detect 的請求格式（base64 圖片）
    DetectResponse,    # POST /detect 的回應格式
    HealthResponse,    # GET  /health 的回應格式
)
from service_api.utils.image_utils import crop_with_mask, pil_to_bytes  # mask 裁切 / 轉 bytes


# ── 接收到的圖片存檔（背景任務，不拖慢回應時間）───────────────────────────────

RECEIVED_IMAGES_DIR = Path(__file__).resolve().parent.parent / "received_images"


def _save_received_image(image_bytes: bytes, filename: str) -> None:
    """
    把收到的原始圖片存檔到 received_images/。

    以 BackgroundTasks 執行：FastAPI 會在回應已送出給客戶端「之後」才呼叫，
    所以存檔耗時完全不會加到 API 回應時間上。
    """
    RECEIVED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix or ".jpg"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = RECEIVED_IMAGES_DIR / f"{ts}{suffix}"
    out_path.write_bytes(image_bytes)


# ── 應用程式生命週期管理 ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan context manager。

    startup（yield 之前）：
      - 建立 BoxDetectionPipeline，載入 YOLO 模型、建立 Gemini client
      - 儲存在 app.state.pipeline，讓所有請求處理函式都能存取

    shutdown（yield 之後）：
      - 目前 YOLO 模型與 Gemini client 無需特別釋放資源
    """
    print("🚀 [startup] 初始化 BoxDetectionPipeline，正在載入 YOLO 模型...")
    app.state.pipeline = BoxDetectionPipeline()  # 建立 pipeline（YOLO 載入在此發生）
    print("✅ [startup] 完成！")
    yield  # ← 應用程式在這裡接受請求，直到 uvicorn 收到終止信號
    print("🛑 [shutdown] 服務關閉")


# ── 建立 FastAPI 應用程式 ─────────────────────────────────────────────────────

app = FastAPI(
    title="Box and Date Recognition API",               # Swagger UI 頁面標題
    description=(
        "紙箱偵測 + 品項比對 + 日期解析的整合辨識 API\n\n"
    ),
    version="1.1.0",
    lifespan=lifespan,  # 使用自訂 lifespan 取代已棄用的 on_event("startup")
)


# ── Middleware：latency 測量 ──────────────────────────────────────────────────
#
# 在每個 response 上附加 server 端時間戳（epoch seconds，float 字串），
# 讓 client 端能把整趟 request 拆成三段：
#   1. client 送出 → server 收到（network，上行）      = X-Server-Recv-Time  減去 client 端送出時間
#   2. server 收到 → 辨識完成（server 端運算）           = X-Server-Inference-Done-Time 減去 X-Server-Recv-Time
#   3. server 回傳 → client 收到（network，下行）        = client 端收到時間 減去 X-Server-Send-Time
#
# X-Server-Inference-Done-Time 只有在路由本身有設定 request.state.t_inference_done
# 時才會出現（目前是 /detect/image），其餘路由（如 /health）只會有 Recv / Send 兩個時間戳，
# 可用來做 client-server 時鐘校正（NTP-like offset 估計）。
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        t_recv = time.time()
        request.state.t_recv = t_recv
        response = await call_next(request)
        t_send = time.time()
        response.headers["X-Server-Recv-Time"] = repr(t_recv)
        response.headers["X-Server-Send-Time"] = repr(t_send)
        t_inference_done = getattr(request.state, "t_inference_done", None)
        if t_inference_done is not None:
            response.headers["X-Server-Inference-Done-Time"] = repr(t_inference_done)
        return response


app.add_middleware(TimingMiddleware)


# ── 路由：健康檢查 ────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["系統"],
    summary="服務狀態檢查",
    description="確認服務狀態是否正常。",
)
def health_check() -> HealthResponse:
    """回傳服務狀態。"""
    return HealthResponse(status="ok")


# ── 路由：偵測（接收 base64 圖片，回傳 JSON，含標注圖 base64）───────────────────

@app.post(
    f"{config.API_PREFIX}/detect/image_base64",
    response_model=DetectResponse,
    tags=["偵測"],
    summary="上傳 Base64 圖片，回傳偵測結果（JSON，含標注圖 base64 與每個紙箱的 mask 座標）",
    description=(
"""
上傳一張圖片的 Base64 編碼字串（`image_base64`），API 會偵測照片中的紙箱，並回傳完整結果的 JSON。

與 `/detect/image` 的差異：這個端點的請求 body 是 JSON（`application/json`），
圖片以 Base64 字串夾帶在 body 裡，而不是用 `multipart/form-data` 上傳檔案。

### 請求參數（JSON body）

- `image_base64`（必填）：圖片的 Base64 編碼字串（JPG / PNG，不含 `data:image/...;base64,` 前綴）
- `include_image`（選填，預設 `false`）：`true` 時回應額外附上 Base64 編碼的標注 JPEG 圖片

### 回應欄位

- `filename`：固定回傳 `"upload.jpg"`（此端點無法取得原始檔名，因為輸入只有 Base64 字串）
- `total_boxes`：偵測到的紙箱總數
- `boxes`：每個紙箱的詳細結果，每筆包含：
  - `box_id`：紙箱編號，從 1 開始
  - `mask`：紙箱輪廓座標點列表 `[[x, y], ...]`，已除以圖片寬高正規化（值域 0.0~1.0）
  - `product`：辨識到的品項 `{brand, name}`，`brand` 無法辨識為 `null`
  - `expiry_date` / `manufacture_date`：`{year, month, day}`，無法解析為 `null`
- `annotated_image_base64`：含標注結果的 JPEG 圖片（Base64 編碼），僅在 `include_image=true` 時才會回傳，否則為 `null`

### 回應範例

```json
{
  "filename": "upload.jpg",
  "total_boxes": 1,
  "boxes": [
    {
      "box_id": 1,
      "mask": [[0.12, 0.08], [0.45, 0.08], [0.45, 0.51], [0.12, 0.51]],
      "product": {"brand": "義美", "name": "洋芋片 青檸口味"},
      "expiry_date": {"year": "2027", "month": "01", "day": "13"},
      "manufacture_date": null
    }
  ],
  "annotated_image_base64": null
}
```

### Python Example

```python
import base64
import requests

with open("input.jpg", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

resp = requests.post(
    "https://logistics2.sstc-aiteam.org/api/v1/detect/image_base64",
    json={"image_base64": image_base64, "include_image": False},
)
resp.raise_for_status()
data = resp.json()
print(f"共偵測到 {data['total_boxes']} 個紙箱")
```"""
    ),
)
def detect_image_base64(
    request: Request,
    body: DetectRequest,
    background_tasks: BackgroundTasks,
) -> DetectResponse:
    """接收 Base64 圖片，偵測並回傳每個紙箱的品名/日期/mask 座標，並依需求附上標注圖片（base64）。"""
    if config.DEBUG:
        print(f"[DEBUG] /detect/image_base64 圖片接收時間：{datetime.now().strftime('%H:%M:%S')}")

    # 解碼 base64（格式錯誤時回傳 400，而不是讓例外往上炸成 500）
    try:
        image_bytes = base64.b64decode(body.image_base64, validate=True)
    except (ValueError, TypeError):
        raise HTTPException(status_code=400, detail="無效的 Base64 編碼。")

    # 驗證圖片格式（同 /detect/image）
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="無效的圖片格式。")

    filename = "upload.jpg"  # base64 輸入沒有原始檔名，固定給一個預設值
    background_tasks.add_task(_save_received_image, image_bytes, filename)

    pipeline: BoxDetectionPipeline = app.state.pipeline
    result = pipeline.run(
        image_bytes      = image_bytes,
        filename         = filename,
        include_image    = body.include_image,  # true 才附標注圖 base64
        print_confidence = True,  # 印出每個命中 box 的 YOLO confidence（同 /detect/show_image）
    )
    # 辨識完成的時間戳，交給 TimingMiddleware 附加到 response header
    # （X-Server-Inference-Done-Time），供 client 端拆分 latency 區段用。
    request.state.t_inference_done = time.time()
    return result



# ── 路由：偵測（回傳 JSON，含標注圖 base64）───────────────────────────────────

@app.post(
    f"{config.API_PREFIX}/detect/image",
    response_model=DetectResponse,
    tags=["偵測"],
    summary="上傳圖片，回傳偵測結果（JSON，含標注圖 base64 與每個紙箱的 mask 座標）",
    description=(
"""
上傳一張圖片（`file`），API 會偵測照片中的紙箱，並回傳完整結果的 JSON。

### 請求參數

- `file`（必填）：圖片檔案，JPG / PNG
- `include_annotated_image`（選填，預設 `false`）：`true` 時回應額外附上 Base64 編碼的標注 JPEG 圖片

### 回應欄位

- `filename`：上傳的原始檔名
- `total_boxes`：偵測到的紙箱總數
- `boxes`：每個紙箱的詳細結果，每筆包含：
  - `box_id`：紙箱編號，從 1 開始
  - `mask`：紙箱輪廓座標點列表 `[[x, y], ...]`，已除以圖片寬高正規化（值域 0.0~1.0）
  - `product`：辨識到的品項 `{brand, name}`，`brand` 無法辨識為 `null`
  - `expiry_date` / `manufacture_date`：`{year, month, day}`，無法解析為 `null`
- `annotated_image_base64`：含標注結果的 JPEG 圖片（Base64 編碼），僅在 `include_annotated_image=true` 時才會回傳，否則為 `null`

### 回應範例

```json
{
  "filename": "input.jpg",
  "total_boxes": 1,
  "boxes": [
    {
      "box_id": 1,
      "mask": [[0.12, 0.08], [0.45, 0.08], [0.45, 0.51], [0.12, 0.51]],
      "product": {"brand": "義美", "name": "洋芋片 青檸口味"},
      "expiry_date": {"year": "2027", "month": "01", "day": "13"},
      "manufacture_date": null
    }
  ],
  "annotated_image_base64": null
}
```

### Python Example

```python
import base64
import requests
from io import BytesIO
from PIL import Image

with open("input.jpg", "rb") as f:
    resp = requests.post(
        "https://logistics2.sstc-aiteam.org/api/v1/detect/image",
        files={"file": f},
        data={"include_annotated_image": "true"},
    )
resp.raise_for_status()
data = resp.json()

print(f"共偵測到 {data['total_boxes']} 個紙箱")
for box in data["boxes"]:
    product = box["product"]
    brand   = product["brand"] if product and product["brand"] else ""
    name    = product["name"] if product else "未知品項"
    expiry  = box["expiry_date"]
    expiry_str = f"{expiry['year']}-{expiry['month']}-{expiry['day']}" if expiry else "無法解析"
    print(f"box_id={box['box_id']} product={brand}{name} expiry_date={expiry_str}")

if data["annotated_image_base64"]:
    image = Image.open(BytesIO(base64.b64decode(data["annotated_image_base64"])))
    image.show()
```"""
    ),
)
def detect_image(
    request: Request,
    file: Annotated[
        UploadFile,
        File(description="要偵測的圖片（JPG / PNG）"),
    ],
    include_annotated_image: Annotated[
        bool,
        Form(description="true 時回應額外附上 Base64 編碼的標注 JPEG 圖片"),
    ] = False,
    background_tasks: BackgroundTasks = None,
) -> DetectResponse:
    """偵測並回傳每個紙箱的品名/日期/mask 座標，並依需求附上標注圖片（base64）。"""
    if config.DEBUG:
        print(f"[DEBUG] /detect/image 圖片接收時間：{datetime.now().strftime('%H:%M:%S')}")

    image_bytes = file.file.read()  # 讀取上傳圖片 bytes

    # 驗證圖片格式（同 /detect）
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="無效的圖片格式。")

    background_tasks.add_task(_save_received_image, image_bytes, file.filename or "unknown.jpg")

    pipeline: BoxDetectionPipeline = app.state.pipeline
    result = pipeline.run(
        image_bytes      = image_bytes,
        filename         = file.filename or "unknown.jpg",
        include_image    = include_annotated_image,  # true 才附標注圖 base64
        print_confidence = True,  # 印出每個命中 box 的 YOLO confidence（同 /detect/show_image）
    )
    # 辨識完成的時間戳，交給 TimingMiddleware 附加到 response header
    # （X-Server-Inference-Done-Time），供 client 端拆分 latency 區段用。
    request.state.t_inference_done = time.time()
    return result


# ── 路由：偵測（直接回傳標注後的 JPEG 圖片 binary）────────────────────────────

@app.post(
    f"{config.API_PREFIX}/detect/show_image",
    include_in_schema=False,
    tags=["偵測"],
    summary="上傳圖片，直接回傳標注後的 JPEG 圖片（binary）",
    description=(
        "辨識流程與 `/detect/image` 相同（YOLO segmentation → Gemini 整圖辨識 → "
        "以 box_2d 中心點比對 YOLO mask），差別在於此端點不回傳 JSON，"
        "而是直接回傳畫好標注結果的 JPEG 圖片本體，"
        "方便直接在瀏覽器開啟或串接到需要圖片檔案的用途。"
    ),
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def detect_show_image(
    file: Annotated[
        UploadFile,
        File(description="要偵測的圖片（JPG / PNG）"),
    ],
) -> Response:
    """偵測並直接回傳標注後的 JPEG 圖片 binary（不回傳 JSON）。"""
    if config.DEBUG:
        print(f"[DEBUG] /detect/show_image 圖片接收時間：{datetime.now().strftime('%H:%M:%S')}")

    image_bytes = file.file.read()  # 讀取上傳圖片 bytes

    # 驗證圖片格式（同 /detect）
    nparr = np.frombuffer(image_bytes, np.uint8)
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="無效的圖片格式。")

    pipeline: BoxDetectionPipeline = app.state.pipeline
    result = pipeline.run(
        image_bytes      = image_bytes,
        filename         = file.filename or "unknown.jpg",
        include_image    = True,  # 一定要畫標注圖，才有東西可以回傳
        print_confidence = True,  # 印出每個命中 box 的 YOLO confidence
    )

    if result.annotated_image_base64 is None:
        # 沒有命中任何 box，pipeline 不會畫圖，直接回傳原圖
        jpg_bytes = image_bytes
    else:
        jpg_bytes = base64.b64decode(result.annotated_image_base64)

    return Response(content=jpg_bytes, media_type="image/jpeg")


# ── 路由：偵測（YOLO segmentation mask 存檔，供人工複核未知品項）──────────────

@app.post(
    f"{config.API_PREFIX}/detect/segment-unknown",
    include_in_schema=False,   # 不顯示在 Swagger UI 文件中，但端點仍可正常呼叫
)
def detect_segment_unknown(
    file: Annotated[
        UploadFile,
        File(description="要偵測的圖片（JPG / PNG）"),
    ],
) -> dict:
    """
    上傳圖片，只執行 YOLO segmentation（不呼叫 Gemini）。
    將每個偵測到的 box mask 去背裁切後存成 JPG，放進
    config.SEGMENTATION_UNKNOWN_DIR，供之後人工複核。
    """
    image_bytes = file.file.read()  # 讀取上傳圖片 bytes

    # 驗證圖片格式（同 /detect）
    nparr   = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise HTTPException(status_code=400, detail="無效的圖片格式。")

    # 只執行 YOLO 偵測（不需要 Gemini）
    pipeline: BoxDetectionPipeline = app.state.pipeline
    detection = pipeline.detector.detect(img_bgr)

    # 準備輸出資料夾（可能是第一次呼叫，資料夾尚未存在）
    save_dir = Path(config.SEGMENTATION_UNKNOWN_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved_files: list[str] = []
    for polygon in detection.polygons:
        cropped_pil = crop_with_mask(img_bgr, polygon)  # 去背裁切
        out_path    = save_dir / f"{uuid.uuid4().hex}.jpg"
        out_path.write_bytes(pil_to_bytes(cropped_pil))
        saved_files.append(out_path.name)

    return {
        "total_boxes": len(detection.polygons),
        "saved_files": saved_files,
    }
