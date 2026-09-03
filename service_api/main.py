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
import uuid                                  # 產生不重複的 mask 檔名
from datetime import datetime                # DEBUG 模式下印出圖片接收時間

import cv2                                   # 驗證上傳圖片是否可解碼
import numpy as np                           # 圖片 bytes 轉 numpy array

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # FastAPI 核心元件
from fastapi.responses import Response       # 直接回傳二進位圖片

from service_api import config               # 所有設定常數
from service_api.pipeline import BoxDetectionPipeline  # 核心業務流程
from service_api.schemas import (
    DetectRequest,     # POST /detect 的請求格式（base64 圖片）
    DetectResponse,    # POST /detect 的回應格式
    HealthResponse,    # GET  /health 的回應格式
)
from service_api.utils.image_utils import crop_with_mask, pil_to_bytes  # mask 裁切 / 轉 bytes


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


# ── 路由：偵測（回傳 JSON）────────────────────────────────────────────────────

@app.post(
    f"{config.API_PREFIX}/detect",
    response_model=DetectResponse,
    tags=["偵測"],
    summary="上傳圖片，回傳每個紙箱的品名與日期（JSON）",
    include_in_schema=False,   # 不顯示在 Swagger UI 文件中，但端點仍可正常呼叫
)
def detect(request: DetectRequest) -> DetectResponse:
    """
    主要偵測端點。

    - 接受 JSON body，圖片以 base64 字串（image_base64）傳入
    - YOLO 偵測所有紙箱 mask，Gemini 對整張圖辨識品名/日期/box_2d，
      再以 box_2d 中心點比對 YOLO mask，命中才保留該筆結果
    - include_image=true 時，回應額外含 base64 編碼的標注圖（JPEG）
    """
    try:
        image_bytes = base64.b64decode(request.image_base64, validate=True)
    except (base64.binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image_base64 不是合法的 Base64 字串。")

    # 基本驗證：確認 bytes 能被 OpenCV 解碼為有效圖片
    nparr = np.frombuffer(image_bytes, np.uint8)   # bytes → numpy uint8 array
    img   = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # 嘗試解碼
    if img is None:
        # cv2.imdecode 回傳 None 代表格式不支援或資料損毀
        raise HTTPException(
            status_code=400,
            detail="無效的圖片格式，請上傳 JPG 或 PNG 檔案。",
        )

    pipeline: BoxDetectionPipeline = app.state.pipeline
    return pipeline.run(
        image_bytes   = image_bytes,
        filename      = "unknown.jpg",
        include_image = request.include_image,
    )


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
  - `product`：辨識到的品項 `{brand_name, product_name}`，`brand_name` 無法辨識為 `null`
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
      "product": {"brand_name": "義美", "product_name": "洋芋片 青檸口味"},
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
        "https://logistics.sstc-aiteam.org/api/v1/detect/image",
        files={"file": f},
        data={"include_annotated_image": "true"},
    )
resp.raise_for_status()
data = resp.json()

print(f"共偵測到 {data['total_boxes']} 個紙箱")
for box in data["boxes"]:
    product = box["product"]
    brand   = product["brand_name"] if product and product["brand_name"] else ""
    name    = product["product_name"] if product else "未知品項"
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
    file: Annotated[
        UploadFile,
        File(description="要偵測的圖片（JPG / PNG）"),
    ],
    include_annotated_image: Annotated[
        bool,
        Form(description="true 時回應額外附上 Base64 編碼的標注 JPEG 圖片"),
    ] = False,
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

    pipeline: BoxDetectionPipeline = app.state.pipeline
    return pipeline.run(
        image_bytes      = image_bytes,
        filename         = file.filename or "unknown.jpg",
        include_image    = include_annotated_image,  # true 才附標注圖 base64
        print_confidence = True,  # 印出每個命中 box 的 YOLO confidence（同 /detect/show_image）
    )


# ── 路由：偵測（直接回傳標注後的 JPEG 圖片 binary）────────────────────────────

@app.post(
    f"{config.API_PREFIX}/detect/show_image",
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
