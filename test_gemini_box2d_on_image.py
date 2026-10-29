"""
test_gemini_box2d_on_image.py — 「圖上畫出 box_2d 矩形與 #id」再讓 Gemini 依 id 辨識品名的實驗腳本
==================================================================================================
與 test_gemini_box2d.py 相同流程，差別在傳給 Gemini 的圖：直接把每個紙箱的矩形與 #id 畫在圖上，
prompt 仍附上「id → box_2d」清單作為輔助：
  1. YOLO segmentation（服務的模型與參數）取得每個紙箱的 polygon，算出外接矩形與中心點
  2. 外接矩形轉成 box_2d：[ymin, xmin, ymax, xmax]，正規化到 0-1000 整數（同服務 gemini_client.py）
  3. 在原圖畫上矩形（顏色依 id）與左上角 #id 標籤，連同「id → box_2d」清單一起傳給 Gemini
     （prompt 的日期、品名規則抄自服務，不叫 Gemini 偵測紙箱；回傳的 id 用 enum 限定）
  4. 依 id 對回紙箱，輸出 result.json，並畫出 YOLO 輪廓 + 中心點品名的 result.jpg

使用方式：
    python3 test_gemini_box2d_on_image.py --image "sample_images/全家/正面-2.jpg" [--out 輸出資料夾]

輸出（預設 results/gemini_box2d_on_image/<圖片檔名>/）：
    result.json  [{id, box_2d, name, manufacturer_date, expiration_date, center, conf, polygon}]
    boxes.jpg    實際傳給 Gemini 的圖：原圖 + box_2d 矩形與 #id
    result.jpg   原圖 + YOLO 輪廓 + 每個紙箱中心點寫上品名
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

from service_api import config
from service_api.services.gemini_client import _DATE_SCHEMA
from test_gemini_box2d import draw_boxes, to_box_2d
from test_gemini_contour import PALETTE, draw_result, fmt_date, polygon_center, yolo_polygons

PROMPT = """你是專業的商品紙箱文字辨識專家。圖中紙箱的位置已經偵測好，共 {n} 個，
每個紙箱已在圖上以彩色矩形框出，矩形左上角的色塊標籤寫著該紙箱的 id（例如 #0、#1）。
各 id 的矩形座標同時以 box_2d 列出：[ymin, xmin, ymax, xmax]，正規化到 0-1000 整數，(0,0) 為左上角、(1000,1000) 為右下角。
紙箱彼此堆疊，矩形可能互相重疊；請以矩形範圍內「最主要、最完整」的那個紙箱為該 id 的紙箱。
矩形框線與 #id 標籤是後製標記，不是箱面印刷內容，辨識文字時請忽略。

{box_list}

請依 id 逐一分析「每一個」紙箱的資訊，每個紙箱各自獨立為清單中一個物件，並遵循以下規則：

1. 日期：有「有效期/到期/EXP」直接取為 expiration_date。
   - 【製造日期+保存期限相加】：僅當同一組日期數字「本身」明確標示「製造日期/生產日期」等字樣，且緊鄰印有對應的保存期限（如10個月），兩者為同一組標示搭配時，才將兩者相加算出 expiration_date。
   - 【無標示的印章/戳印數字】：若箱面某處只是一串戳印/噴印的日期數字（無任何文字前綴，例如僅印「20241226」+批號），即使箱面「其他地方」另外印有一般規格性質的保存期限文字（如「保存期限:3年」，屬於箱體固定印刷規格說明、並非緊鄰該戳印日期的搭配標示），仍應直接將該戳印日期本身視為 expiration_date，不可自行當作製造日期並加上保存期限重新計算；manufacturer_date 維持查無資訊。
   - 查無日期則 manufacturer_date/expiration_date 填 null。統一格式 YYYY-MM-DD。

2. product_name：必須是箱面實際印刷文字的連續子字串，不可改寫/翻譯/增字/調換順序，逐字核對確保與原文一致（含形近字如「菲」非「非」）。
   - 取「公司/品牌名稱＋商品類別或品名」的組合，長度以完整表達商品類型為原則，不因求簡短而截斷語意（如需保留「紙湯杯」不可只留「環保」）；不加「袋裝」「箱」等未直接相連的字。
   - 有口味/風味/規格等細分資訊（如「牛肉蔬菜風味」）必須完整包含，不可只取品牌/大類而省略。
   - 箱面若印有公司/品牌 Logo（如橢圓色塊搭配簡短文字，例：「華元」），即使 Logo 與品名文字位置分開，product_name 仍須以該 Logo 文字為前綴（例：「華元 真魷味 紅燒口味」）；【絕對不可省略】品牌文字是 product_name 的必要組成，不可因加速輸出而簡化省略。
   - 【書法/藝術字體品牌】：若品牌文字採用書法/毛筆/藝術字體印刷（筆畫較粗、連筆、非標準印刷字形），或紙箱拍攝角度傾斜/旋轉導致文字非水平，仍須耐心逐筆辨識，不可因字體特殊或角度傾斜而略過、猜測形似字、或直接省略；可留意品牌文字旁常伴隨的®/™小圓圈符號來確認該處為品牌標示。
   - 若印有品名/口味表格：(a) 某欄位有記號（圓點）標示實際內容物，則取「品牌＋該欄品名」（例：MOS BURGER 表格「蕃茄燉雞肉」欄有記號，則為「MOS BURGER 蕃茄燉雞肉」）；(b) 表格字跡模糊無法逐字確認，但有清楚可辨識的共通關鍵字（如各列都以「冰塊」結尾），則取「品牌＋該關鍵字」，模糊到無法確定的修飾字（如「袋裝/鋼製」）絕不可自行猜測，寧可精簡。

3. 紙箱已由圖上的矩形與 #id 指定：每個 id 都要回傳一筆，不可新增清單以外的 id、不可合併或拆分；
   若該 id 的紙箱被遮擋、只露出一小部分或字跡模糊，無法從它自己的箱面讀出品名，product_name 填 null，不可從相鄰紙箱推測。

輸出所有 id 紙箱的結構化物件陣列。"""


def ask_gemini(image_jpg: bytes, boxes: list[list[int]]) -> list[dict]:
    box_list = "\n".join(f"- id {i}: {b}" for i, b in enumerate(boxes))
    prompt = PROMPT.format(n=len(boxes), box_list=box_list)
    print(f"[PROMPT]\n{prompt}\n")
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                # enum 只接受字串，id 以字串回傳再轉回整數
                "id": {"type": "STRING", "enum": [str(i) for i in range(len(boxes))]},
                "product_name": {"type": "STRING", "nullable": True},
                "manufacturer_date": _DATE_SCHEMA,
                "expiration_date": _DATE_SCHEMA,
            },
            "required": ["id", "product_name", "manufacturer_date", "expiration_date"],
        },
    }

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=image_jpg, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    usage = response.usage_metadata
    print(f"[GEMINI] tokens: prompt={usage.prompt_token_count} thoughts={usage.thoughts_token_count or 0} "
          f"output={usage.candidates_token_count or 0} total={usage.total_token_count}")
    return json.loads(response.text)


def main() -> None:
    parser = argparse.ArgumentParser(description="圖上畫 box_2d 矩形與 #id → Gemini 依 id 辨識品名")
    parser.add_argument("--image", required=True, help="紙箱圖片")
    parser.add_argument("--out",   help="輸出資料夾（預設 results/gemini_box2d_on_image/<圖片檔名>）")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"無法讀取圖片：{image_path}")
    h, w = img.shape[:2]
    out_dir = Path(args.out) if args.out else config.ROOT_DIR / "results" / "gemini_box2d_on_image" / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    polygons, confs = yolo_polygons(img)
    if not polygons:
        raise SystemExit("YOLO 沒有偵測到紙箱")
    boxes = [to_box_2d(p, w, h) for p in polygons]
    print(f"圖片：{image_path}  紙箱：{len(polygons)} 個 → 輸出到 {out_dir}")

    boxed = draw_boxes(img, boxes)
    cv2.imwrite(str(out_dir / "boxes.jpg"), boxed, [cv2.IMWRITE_JPEG_QUALITY, 95])
    image_jpg = cv2.imencode(".jpg", boxed, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    t0 = time.perf_counter()
    answers = ask_gemini(image_jpg, boxes)
    print(f"[GEMINI] {time.perf_counter() - t0:.2f}s")

    by_id: dict[int, dict] = {}
    for a in answers:
        i = int(a["id"])
        if i not in by_id:
            by_id[i] = a
        else:
            print(f"⚠️ 忽略 Gemini 回傳的重複 id：{a}")
    missing = [i for i in range(len(polygons)) if i not in by_id]
    if missing:
        print(f"⚠️ Gemini 沒回這些 id：{missing}")

    result = []
    for i, (p, c, b) in enumerate(zip(polygons, confs, boxes)):
        a = by_id.get(i, {})
        result.append({
            "id": i,
            "box_2d": b,
            "name": a.get("product_name"),
            "manufacturer_date": a.get("manufacturer_date"),
            "expiration_date": a.get("expiration_date"),
            "center": [round(v, 1) for v in polygon_center(p)],
            "conf": round(c, 3),
            "polygon": np.round(p, 1).tolist(),
            "color": PALETTE[i % len(PALETTE)][0],   # 只用來畫結果圖
        })
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    draw_result(img, result).save(out_dir / "result.jpg", quality=95)

    for r in result:
        print(f"  #{r['id']:<2} box_2d={r['box_2d']}  name={r['name']}  "
              f"mfg={fmt_date(r['manufacturer_date'])}  exp={fmt_date(r['expiration_date'])}")
    print(f"→ {out_dir / 'result.json'}、{out_dir / 'result.jpg'}")


if __name__ == "__main__":
    main()
