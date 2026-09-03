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

PROMPT = """你是專業的商品紙箱文字辨識與空間定位專家。分析圖片中所有紙箱，每個紙箱各自獨立為清單中一個物件，並遵循以下規則：

1. 日期：有「有效期/到期/EXP」直接取為 expiration_date。
   - 【製造日期+保存期限相加】：僅當同一組日期數字「本身」明確標示「製造日期/生產日期」等字樣，且緊鄰印有對應的保存期限（如10個月），兩者為同一組標示搭配時，才將兩者相加算出 expiration_date。
   - 【無標示的印章/戳印數字】：若箱面某處只是一串戳印/噴印的日期數字（無任何文字前綴，例如僅印「20241226」+批號），即使箱面「其他地方」另外印有一般規格性質的保存期限文字（如「保存期限:3年」，屬於箱體固定印刷規格說明、並非緊鄰該戳印日期的搭配標示），仍應直接將該戳印日期本身視為 expiration_date，不可自行當作製造日期並加上保存期限重新計算；manufacturer_date 維持查無資訊。
   - 查無日期則 manufacturer_date/expiration_date 填 "未知"。統一格式 YYYY-MM-DD。

2. product_name：必須是箱面實際印刷文字的連續子字串，不可改寫/翻譯/增字/調換順序，逐字核對確保與原文一致（含形近字如「菲」非「非」）。
   - 取「公司/品牌名稱＋商品類別或品名」的組合，長度以完整表達商品類型為原則，不因求簡短而截斷語意（如需保留「紙湯杯」不可只留「環保」）；不加「袋裝」「箱」等未直接相連的字。
   - 有口味/風味/規格等細分資訊（如「牛肉蔬菜風味」）必須完整包含，不可只取品牌/大類而省略。
   - 箱面若印有公司/品牌 Logo（如橢圓色塊搭配簡短文字，例：「華元」），即使 Logo 與品名文字位置分開，product_name 仍須以該 Logo 文字為前綴（例：「華元 真魷味 紅燒口味」）；【絕對不可省略】品牌文字是 product_name 的必要組成，不可因加速輸出而簡化省略。
   - 【書法/藝術字體品牌】：若品牌文字採用書法/毛筆/藝術字體印刷（筆畫較粗、連筆、非標準印刷字形），或紙箱拍攝角度傾斜/旋轉導致文字非水平，仍須耐心逐筆辨識，不可因字體特殊或角度傾斜而略過、猜測形似字、或直接省略；可留意品牌文字旁常伴隨的®/™小圓圈符號來確認該處為品牌標示。
   - 若印有品名/口味表格：(a) 某欄位有記號（圓點）標示實際內容物，則取「品牌＋該欄品名」（例：MOS BURGER 表格「蕃茄燉雞肉」欄有記號，則為「MOS BURGER 蕃茄燉雞肉」）；(b) 表格字跡模糊無法逐字確認，但有清楚可辨識的共通關鍵字（如各列都以「冰塊」結尾），則取「品牌＋該關鍵字」，模糊到無法確定的修飾字（如「袋裝/鋼製」）絕不可自行猜測，寧可精簡。

3. 紙箱偵測：只框輪廓完整清晰可見者，被遮擋大半或只露極小一角則忽略；同一實體紙箱僅一筆資料，不因文字分散於多面而拆成多筆；裝飾圖案/警示圖示/條碼不可單獨當作紙箱。

4. box_2d：[ymin, xmin, ymax, xmax]，正規化到 0-1000 整數，(0,0)為左上角、(1000,1000)為右下角，須貼齊紙箱可見輪廓。

輸出所有紙箱的結構化物件陣列。"""

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

        boxes = json.loads(response.text)
        for box in boxes:
            print(f"[GEMINI] product_name={box.get('product_name')}")
        return boxes
