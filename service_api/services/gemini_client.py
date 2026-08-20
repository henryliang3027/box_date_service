"""
services/gemini_client.py — Gemini 紙箱辨識封裝
==================================================
呼叫 Gemini 對「整張圖片」做推論，一次取得所有紙箱的：
  - product_name（品名）
  - manufacturer_date / expiration_date（製造日期 / 有效日期）
  - box_2d（邊界框，[ymin, xmin, ymax, xmax]，正規化到 0-1000）

box_2d 只用來算出每個紙箱的「中心點」，供 pipeline.py 拿去和
YOLO segmentation mask 做比對（中心點落在哪個 mask 內，就採用該 mask）。
"""

import json
import mimetypes

from google import genai
from google.genai import types

from service_api import config

PROMPT = """你是一個專業的商品紙箱文字辨識與空間定位專家。請分析圖片中所有的紙箱，並將每一個紙箱分別獨立為清單中的一個物件。

請嚴格遵循以下解析規則：

1. 日期與保存期限推理邏輯：
   - 【直接提取】：若印有「有效日期 / 到期日 / EXP」，直接提取該日期作為 expiration_date。
   - 【公式計算】：若同時印有「製造日期」與「保存期限」（例如：保存期限 10 個月），請自動將製造日期加上保存期限，計算出最終的有效日期作為 expiration_date。
   - 【預設規則】：若僅看到一串日期數字且沒有任何前綴說明（如未標明製造或有效），一律直接視為 expiration_date。
   - 【無資訊處理】：若無法獲知製造日期，manufacturer_date 請填寫 "未知"；若無有效日期資訊則填寫 "未知"。所有日期格式請統一轉換為 YYYY-MM-DD。

2. 品名擷取規範：
   - product_name 必須是箱面上實際印刷文字的「連續子字串」，不可自行改寫、翻譯、增字或重組詞語順序。
   - 從箱面所有文字中，挑選最適合代表該商品的一段文字，
     例如公司名稱＋商品類別的組合（如「琦美製冰」），而非完整公司全名（如「琦美製冰股份有限公司」），
     也不要額外加上「袋裝」「箱」等箱面上没有直接連在一起的字詞。
   - 長度以能完整表達商品類型為原則，不要為了縮短而截斷成語意不完整的詞（例如不可只留「環保」而捨去後面的「紙湯杯」）。
   - 若箱面印有口味／風味／規格等細分資訊（例如「牛肉蔬菜風味」「鮮蝦魚板風味」「韓式泡菜風味」），
     必須將該資訊完整包含在 product_name 中，不可只取品牌或品項大類而省略口味/風味文字。
   - 若箱面有明確商品名稱（如印刷的商品標題），優先直接採用該商品名稱作為 product_name，且同樣要包含口味/風味等細分資訊。
   - 逐字核對：輸出前請將擷取的每個字與圖片中的印刷文字逐字比對，避免看錯字形相近的字（例如「菲」誤讀為「非」、「己」誤讀為「已」等），確保 100% 與原文一致。

3. 紙箱偵測與去重規範：
   - 只框出輪廓完整、清晰可見的紙箱；若紙箱被遮擋、堆疊物遮住大半、或只有極小一角入鏡，一律忽略不列入。
   - 每一個實體紙箱只能對應「一筆」資料，即使該紙箱的文字或標籤分散在多個側面/區域，也必須合併為單一物件，不可因文字分佈不同而拆成多筆重複偵測。
   - 不要將箱子外觀上的裝飾圖案、警示圖示、條碼等局部區域單獨當作一個紙箱物件。

4. 空間定位規範：
   - 請框出每個紙箱完整外觀的邊界框 box_2d，格式為 [ymin, xmin, ymax, xmax]。
   - 座標數值須正規化到 0-1000 的整數範圍，(0, 0) 代表圖片左上角，(1000, 1000) 代表圖片右下角。
   - 邊界框必須盡量貼齊該紙箱可見的完整輪廓邊緣。

請依據上述規則，輸出包含所有紙箱資訊的結構化物件陣列。"""

_DATE_SCHEMA = {
    "type": "OBJECT",
    "nullable": True,
    "description": "無法解析出日期時回傳 null",
    "properties": {
        "year": {"type": "STRING", "description": "四位數年份，例如 2027"},
        "month": {"type": "STRING", "description": "兩位數月份，例如 01"},
        "day": {"type": "STRING", "description": "兩位數日期，例如 13"},
    },
    "required": ["year", "month", "day"],
}

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "product_name": {"type": "STRING"},
            "manufacturer_date": _DATE_SCHEMA,
            "expiration_date": _DATE_SCHEMA,
            "box_2d": {
                "type": "ARRAY",
                "description": "紙箱邊界框 [ymin, xmin, ymax, xmax]，正規化到 0-1000 整數",
                "items": {"type": "INTEGER"},
            },
        },
        "required": [
            "product_name",
            "manufacturer_date",
            "expiration_date",
            "box_2d",
        ],
    },
}


class GeminiBoxClient:
    """
    封裝對 Gemini 的整張圖推論呼叫。

    使用方式：
        client = GeminiBoxClient()
        boxes  = client.detect(image_bytes, filename="input.jpg")
        # boxes: [{"product_name", "manufacturer_date", "expiration_date", "box_2d"}, ...]
    """

    def __init__(self) -> None:
        self._client = genai.Client(api_key=config.GEMINI_API_KEY)

    def detect(self, image_bytes: bytes, filename: str = "image.jpg") -> list[dict]:
        """
        對整張圖片執行 Gemini 推論，回傳所有紙箱的結構化資訊列表。

        Args:
            image_bytes: 上傳圖片的原始 bytes
            filename:    原始檔名（僅用來猜測 mime type）

        Returns:
            list[dict]，每筆包含 product_name / manufacturer_date /
            expiration_date / box_2d
        """
        mime_type = mimetypes.guess_type(filename)[0] or "image/jpeg"

        response = self._client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                PROMPT,
            ],
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
                temperature=0,
                response_mime_type="application/json",
                response_schema=SCHEMA,
            ),
        )

        return json.loads(response.text)
