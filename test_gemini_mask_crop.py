"""
test_gemini_mask_crop.py — 「依 YOLO mask 切出每個紙箱，逐張送 Gemini」辨識品名的實驗腳本
==========================================================================================
每個紙箱各自切成一張獨立的圖，Gemini 一次只看一個紙箱，不需要靠座標或標記指定是哪一箱：
  1. YOLO segmentation（服務的模型與參數）取得每個紙箱的 polygon，算出中心點
  2. 依 polygon 切出每個紙箱：取外接矩形（外擴 pad 像素），矩形內 polygon 以外的像素塗白，存成 crops/crop_<id>.jpg
  3. 每張切圖各自呼叫一次 Gemini（一般 API，多執行緒同時送出），回傳該紙箱的品名與日期
     （prompt 的日期、品名規則抄自服務，不叫 Gemini 偵測紙箱）
  4. 依 id 對回紙箱，輸出 result.json，並畫出 YOLO 輪廓 + 中心點品名的 result.jpg

使用方式：
    python3 test_gemini_mask_crop.py --image "sample_images/全家/正面-2.jpg" [--out 輸出資料夾] [--workers 8]

輸出（預設 results/gemini_mask_crop/<圖片檔名>/）：
    crops/crop_<id>.jpg  送給 Gemini 的紙箱切圖
    result.json          [{id, name, manufacturer_date, expiration_date, center, conf, polygon, crop, seconds}]
    result.jpg           原圖 + YOLO 輪廓 + 每個紙箱中心點寫上品名
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from google import genai
from google.genai import types

from service_api import config
from service_api.services.gemini_client import _DATE_SCHEMA
from test_gemini_batch import crop_by_polygon
from test_gemini_contour import PALETTE, draw_result, fmt_date, polygon_center, yolo_polygons

PROMPT = """你是專業的商品紙箱文字辨識專家。這張圖是從照片中依輪廓切下來的「單一個」紙箱，
紙箱輪廓以外的區域已塗成白色，不屬於這個紙箱。請分析這個紙箱，並遵循以下規則：

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

3. 紙箱已切好，固定只回傳這一個紙箱的一筆物件：只看這個紙箱本身的箱面文字，切圖邊緣可能殘留相鄰紙箱的一小部分，不可採用；
   若紙箱被遮擋、只露出一小部分或字跡模糊，無法從它自己的箱面讀出品名，product_name 填 null，不可猜測。

輸出這個紙箱的結構化物件。"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "product_name": {"type": "STRING", "nullable": True},
        "manufacturer_date": _DATE_SCHEMA,
        "expiration_date": _DATE_SCHEMA,
    },
    "required": ["product_name", "manufacturer_date", "expiration_date"],
}


def ask_gemini(client: genai.Client, crop: np.ndarray) -> tuple[dict, float, int]:
    """送一張紙箱切圖，回傳（辨識結果, 耗時秒數, total tokens）。"""
    jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    t0 = time.perf_counter()
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[types.Part.from_bytes(data=jpg, mime_type="image/jpeg"), PROMPT],
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
            temperature=0,
            response_mime_type="application/json",
            response_schema=SCHEMA,
        ),
    )
    return json.loads(response.text), time.perf_counter() - t0, response.usage_metadata.total_token_count


def main() -> None:
    parser = argparse.ArgumentParser(description="依 YOLO mask 切出每個紙箱 → 逐張送 Gemini 辨識品名")
    parser.add_argument("--image",   required=True, help="紙箱圖片")
    parser.add_argument("--out",     help="輸出資料夾（預設 results/gemini_mask_crop/<圖片檔名>）")
    parser.add_argument("--workers", type=int, default=8, help="同時送出的 Gemini 請求數（預設 8）")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"無法讀取圖片：{image_path}")
    out_dir = Path(args.out) if args.out else config.ROOT_DIR / "results" / "gemini_mask_crop" / image_path.stem
    crop_dir = out_dir / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    polygons, confs = yolo_polygons(img)
    if not polygons:
        raise SystemExit("YOLO 沒有偵測到紙箱")
    print(f"圖片：{image_path}  紙箱：{len(polygons)} 個 → 輸出到 {out_dir}")

    crops = [crop_by_polygon(img, p) for p in polygons]
    for i, c in enumerate(crops):
        cv2.imwrite(str(crop_dir / f"crop_{i}.jpg"), c, [cv2.IMWRITE_JPEG_QUALITY, 95])

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(ask_gemini, client, c) for c in crops]
        answers = []
        for i, f in enumerate(futures):
            try:
                answers.append(f.result())
            except Exception as e:
                print(f"⚠️ #{i} Gemini 呼叫失敗：{e}")
                answers.append(({}, None, 0))
    print(f"[GEMINI] {len(crops)} 次呼叫，總耗時 {time.perf_counter() - t0:.2f}s，"
          f"tokens 合計 {sum(a[2] or 0 for a in answers)}")

    result = []
    for i, (p, c, crop, (a, sec, _)) in enumerate(zip(polygons, confs, crops, answers)):
        result.append({
            "id": i,
            "name": a.get("product_name"),
            "manufacturer_date": a.get("manufacturer_date"),
            "expiration_date": a.get("expiration_date"),
            "center": [round(v, 1) for v in polygon_center(p)],
            "conf": round(c, 3),
            "polygon": np.round(p, 1).tolist(),
            "crop": f"crops/crop_{i}.jpg",
            "seconds": round(sec, 2) if sec is not None else None,
            "color": PALETTE[i % len(PALETTE)][0],   # 只用來畫結果圖
        })
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    draw_result(img, result).save(out_dir / "result.jpg", quality=95)

    for r, crop in zip(result, crops):
        print(f"  #{r['id']:<2} crop={crop.shape[1]}x{crop.shape[0]}  {r['seconds']}s  name={r['name']}  "
              f"mfg={fmt_date(r['manufacturer_date'])}  exp={fmt_date(r['expiration_date'])}")
    print(f"→ {out_dir / 'result.json'}、{out_dir / 'result.jpg'}、{crop_dir}/")


if __name__ == "__main__":
    main()
