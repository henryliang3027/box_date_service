"""
schemas.py — API 請求 / 回應的資料格式定義
==========================================
使用 Pydantic 定義所有 API 的輸入輸出型別。
FastAPI 會自動根據這些類別做型別驗證，並產生 Swagger 文件。

新增欄位時只需在這裡修改，Swagger (/docs) 會自動更新。
"""

from typing import Optional          # 表示可為 None 的型別

from pydantic import BaseModel, Field  # Pydantic 資料模型基礎類別


# ── 子資料結構 ────────────────────────────────────────────────────────────────

class DateInfo(BaseModel):
    """解析後的日期欄位（三個字串欄位）。"""
    year:  str = Field(description="西元年（4 位字串），例如 '2027'")
    month: str = Field(description="月份（補零 2 位字串），例如 '01'")
    day:   str = Field(description="日期（補零 2 位字串），例如 '13'")


class ProductInfo(BaseModel):
    """Gemini 辨識到的產品資訊。"""
    brand_name:   Optional[str] = Field(None, description="品牌／公司名稱，例如 '義美'；無法辨識為 null")
    product_name: str = Field(description="品名，例如 '洋芋片 青檸口味'")


# ── 每個 Box 的結果 ───────────────────────────────────────────────────────────

class BoxResult(BaseModel):
    """單一紙箱的偵測與辨識結果。

    mask 採用 YOLO segmentation 得到的紙箱輪廓；product / 日期資訊則採用
    Gemini 對整張圖推論的結果 —— 只有當 Gemini 該筆結果的 box_2d 中心點
    落在某個 YOLO mask 內時，才會保留為一筆 BoxResult。
    """
    box_id:           int                    = Field(description="Box 編號，從 1 開始計算")
    mask:             list[list[float]]      = Field(description="YOLO mask 輪廓座標點列表，每點為 [x, y]，已除以圖片寬高正規化（值域 0.0~1.0）")
    product:          Optional[ProductInfo]  = Field(None, description="Gemini 辨識到的產品，未知品項為 null")
    expiry_date:      Optional[DateInfo]     = Field(None, description="有效日期（Gemini），無法解析為 null")
    manufacture_date: Optional[DateInfo]     = Field(None, description="製造日期（Gemini），無法解析為 null")


# ── 偵測請求 ──────────────────────────────────────────────────────────────────

class DetectRequest(BaseModel):
    """POST /api/v1/detect 的請求格式。"""
    image_base64:  str            = Field(description="圖片的 Base64 編碼字串（JPG / PNG，不含 data URL 前綴）")
    include_image: bool           = Field(False, description="true 時回應額外附上 Base64 JPEG 標注圖")


# ── 整張圖的偵測回應 ──────────────────────────────────────────────────────────

class DetectResponse(BaseModel):
    """POST /api/v1/detect 的完整回應格式。"""
    filename:                str             = Field(description="上傳的原始檔名")
    total_boxes:             int             = Field(description="YOLO mask 與 Gemini box_2d 皆命中的紙箱總數")
    boxes:                   list[BoxResult] = Field(description="每個紙箱的詳細結果列表")
    annotated_image_base64:  Optional[str]   = Field(
        None,
        description="含標注結果的圖片（Base64 編碼的 JPEG），只在 include_image=true 時才有值"
    )


# ── 健康檢查回應 ──────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """GET /health 的回應格式。"""
    status: str = Field(description="'ok' 代表服務正常運作")
