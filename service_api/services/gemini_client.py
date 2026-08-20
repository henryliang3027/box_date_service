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

PROMPT = """你是專業的商品紙箱辨識與空間定位專家。請找出圖片中每一個紙箱，各自獨立輸出為清單中的一筆物件，並依下列原則作答：

1. 日期推理：
   - 若印有「有效日期／到期日／EXP」，直接採用為 expiration_date。
   - 若同時印有「製造日期」與「保存期限」，將兩者相加推算出 expiration_date。
   - 若只看到一串日期數字、沒有任何前綴說明，一律視為 expiration_date。
   - 無法判斷的日期欄位回傳 null；有日期時，year/month/day 皆為字串。

2. 品名擷取：
   - product_name 必須是箱面實際印刷文字的連續子字串，依照原本印刷順序擷取，不可改寫、翻譯、增字或調換文字順序。
   - 只擷取「用來識別這是什麼商品」的核心文字，即品牌、品名、口味/風味/劑量等能區分商品差異的資訊，
     完整擷取這些核心文字、不要為了求簡短而漏掉其中任何一段；但不要納入容量、數量、包裝規格、銷售條件等與辨識商品無關的說明文字。
   - 若同一箱面有多層可用文字（例如公司登記全名 vs. 簡短品牌、完整商品標題 vs. 局部描述），優先採用最貼近「商品標題／品名標示」的那一段完整文字，
     而非冗長的公司登記全名。
   - 【差異化資訊必列】：若圖片中出現多個相同品牌但款式/口味/劑量不同的商品，product_name 必須包含足以彼此區分的差異化文字，
     絕對不可只輸出品牌名稱（例如多個紙箱都印有「來一客」但口味不同時，每一筆都必須各自帶出對應的口味文字，不可全部只寫「來一客」）。
   - 逐字核對：輸出前將擷取文字與圖片中印刷文字逐一比對，確保每個字元、順序都與原文完全一致、沒有誤讀或遺漏。

3. 紙箱偵測：
   - 盡量找出圖片中所有能被辨識為紙箱的物件，包含畫面邊緣被裁切、只露出局部側面或一角的紙箱，
     只要可見部分足以判斷它是一個獨立紙箱（能看出箱體邊緣、稜角，或帶有部分文字/圖案）即列入。
   - 若某個候選區域毫無可辨識特徵（僅為模糊色塊、極小殘影，不足以確認是獨立紙箱），或幾乎完全被其他物體遮蔽，則不列入，
     也不可為了湊數把同一片不明區域拆成多筆猜測性資料。
   - 同一個實體紙箱只能輸出一筆資料，不可因文字或標籤分散在不同側面而拆成多筆；也不要把裝飾圖案、警示圖示、條碼等局部區域單獨當作紙箱。
   - 若紙箱上沒有任何可辨識文字，product_name 填「無法辨識」。

4. 空間定位：
   - 以 box_2d: [ymin, xmin, ymax, xmax] 表示每個紙箱的邊界框，座標正規化到 0-1000 整數，
     (0, 0) 為圖片左上角、(1000, 1000) 為右下角，邊界框應盡量貼齊該紙箱可見輪廓。

請依此輸出所有紙箱的結構化清單。"""

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
