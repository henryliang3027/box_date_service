# Box and Date Recognition API（YOLO + Gemini 版）

紙箱偵測（YOLO segmentation）+ 品名 / 日期辨識（Gemini）的整合 API。
本專案 fork 自 `service_template`，將原本的 GLM-OCR + 品名資料庫比對流程，
換成 Gemini 對整張圖直接推論品名與日期，並用 box_2d 中心點比對 YOLO mask 做二次驗證。

## 辨識流程

1. YOLO segmentation 偵測整張圖中所有紙箱，取得每個紙箱的 mask polygon
2. Gemini（`API_Test/gemini_ocr.py` 的 prompt/schema）對整張圖推論，取得每個紙箱的
   `product_name` / `manufacturer_date` / `expiration_date` / `box_2d`
3. 將每筆 Gemini 結果的 `box_2d` 換算成中心點座標，判斷該中心點是否落在某個 YOLO mask 內
   - 落在 mask 內 → 保留該 mask，以 Gemini 的品名/日期組成一筆結果
   - 沒有落在任何 mask 內 → 捨棄
4. 將 Gemini 辨識出的 `product_name` 與 `box_name.json` 資料庫（brand/name/keywords）做
   模糊比對，取得正確、統一的 `brand_name` / `product_name`；比對不到時 `brand_name` 為
   `null`，`product_name` 沿用 Gemini 的原始文字
5. `total_boxes` = YOLO mask 與 Gemini box_2d 皆命中的紙箱數

## 回應格式重點（相較舊版的差異）

- `mask`：採用命中的 YOLO segmentation mask（而非 Gemini 的 box_2d）
- `confidence` 欄位已移除
- `product` 包含 `brand_name`（資料庫比對到的品牌，查無則為 `null`）與 `product_name`
  （資料庫比對到的品名，查無則沿用 Gemini 原始辨識文字）
- `expiry_date` / `manufacture_date` 都來自 Gemini 的推論結果

## 目錄結構

```
box_date_service/
├── box_name.json                ← 紙箱品名資料庫（brand/name/keywords，供模糊比對）
├── service_api/
│   ├── main.py                ← FastAPI 入口、路由定義
│   ├── config.py               ← 設定常數（YOLO 路徑、Gemini API Key/Model、BOX_DB_PATH 等）
│   ├── schemas.py               ← API 請求/回應的 Pydantic 格式
│   ├── pipeline.py              ← 核心流程協調者（YOLO 偵測 → Gemini 整圖推論 → mask 比對 → 品名資料庫比對）
│   ├── services/
│   │   ├── detector.py          ← YOLO 封裝
│   │   ├── gemini_client.py     ← Gemini 整圖推論封裝
│   │   └── box_matcher.py       ← 品名資料庫模糊比對封裝
│   └── utils/
│       └── image_utils.py       ← 影像處理工具（裁切、繪圖）
└── models/yolo_model/box_segmentation/  ← YOLO 權重
```

## 前置條件

### 1. 設定 Gemini API Key

在專案根目錄建立 `.env` 檔案，內容如下（將 `your-api-key` 換成你自己的 Gemini API Key）：

```
GEMINI_API_KEY=your-api-key
```


### 2. 安裝依賴

```bash
pip install fastapi uvicorn opencv-python-headless numpy pillow ultralytics google-genai python-multipart python-dotenv
```

### 3. 確認 YOLO 模型存在

```
models/yolo_model/box_segmentation/best_26x_seg_20260702.pt
```

## 啟動 API 服務

```bash
uvicorn service_api.main:app --host 0.0.0.0 --port 8080 --reload
```

啟動後開啟 http://localhost:8080/docs 可看 Swagger UI。

## API 端點

- `GET  /health` — 健康檢查
- `POST /api/v1/detect/image` — 上傳圖片，回傳 JSON（品名/日期/mask，選填標注圖 base64）
- `POST /api/v1/detect/show_image` — 上傳圖片，直接回傳畫好標注結果的 JPEG binary
- `POST /api/v1/detect/segment-unknown` — 只跑 YOLO segmentation，裁切去背存檔（供人工複核）
