"""
test_gemini_box2d.py — 「原圖 + 每個紙箱的 box_2d 座標」讓 Gemini 依 id 辨識品名的實驗腳本
==========================================================================================
圖上不畫任何標記，改用 Gemini 訓練時就熟悉的 box_2d 座標格式告訴它每個紙箱在哪裡：
  1. YOLO segmentation（服務的模型與參數）取得每個紙箱的 polygon，算出外接矩形與中心點
  2. 外接矩形轉成 box_2d：[ymin, xmin, ymax, xmax]，正規化到 0-1000 整數（同服務 gemini_client.py）
  3. 原圖 + 「id → box_2d」清單一起傳給 Gemini，請它依 id 回品名與日期
     （prompt 沿用服務的規則，去掉 box_2d 偵測；回傳的 id 用 enum 限定）
  4. 依 id 對回紙箱，輸出 result.json，並畫出 YOLO 輪廓 + 中心點品名的 result.jpg

使用方式：
    python3 test_gemini_box2d.py --image "sample_images/全家/正面-2.jpg" [--out 輸出資料夾]

輸出（預設 results/gemini_box2d/<圖片檔名>/）：
    result.json  [{id, box_2d, name, manufacturer_date, expiration_date, center, conf, polygon}]
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
from test_gemini_contour import PALETTE, _RULES, draw_result, fmt_date, polygon_center, yolo_polygons

PROMPT = """你是專業的商品紙箱文字辨識專家。圖中紙箱的位置已經偵測好，共 {n} 個，
每個紙箱以 box_2d 表示：[ymin, xmin, ymax, xmax]，正規化到 0-1000 整數，(0,0) 為左上角、(1000,1000) 為右下角。
紙箱彼此堆疊，矩形可能互相重疊；請以矩形範圍內「最主要、最完整」的那個紙箱為該 id 的紙箱。

{box_list}

請依 id 逐一分析「每一個」紙箱的資訊，每個紙箱各自獨立為清單中一個物件，並遵循以下規則：

{rules}

4. 紙箱已由上面的 id 與 box_2d 指定：每個 id 都要回傳一筆，不可新增清單以外的 id、不可合併或拆分；
   依第 3 條應忽略的紙箱（被遮擋大半、只露極小一角）仍要回傳一筆，product_name 填 "未知"、日期填 null，不可從相鄰紙箱推測。

輸出所有 id 紙箱的結構化物件陣列。"""


def to_box_2d(poly: np.ndarray, w: int, h: int) -> list[int]:
    """polygon 外接矩形 → [ymin, xmin, ymax, xmax]，正規化到 0-1000 整數。"""
    x, y, bw, bh = cv2.boundingRect(np.round(poly).astype(np.int32))
    return [round(y / h * 1000), round(x / w * 1000), round((y + bh) / h * 1000), round((x + bw) / w * 1000)]


def ask_gemini(image_jpg: bytes, boxes: list[list[int]]) -> list[dict]:
    box_list = "\n".join(f"- id {i}: {b}" for i, b in enumerate(boxes))
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                # enum 只接受字串，id 以字串回傳再轉回整數
                "id": {"type": "STRING", "enum": [str(i) for i in range(len(boxes))]},
                "product_name": {"type": "STRING"},
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
            PROMPT.format(n=len(boxes), box_list=box_list, rules=_RULES),
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
    parser = argparse.ArgumentParser(description="原圖 + box_2d 座標 → Gemini 依 id 辨識品名")
    parser.add_argument("--image", required=True, help="紙箱圖片")
    parser.add_argument("--out",   help="輸出資料夾（預設 results/gemini_box2d/<圖片檔名>）")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"無法讀取圖片：{image_path}")
    h, w = img.shape[:2]
    out_dir = Path(args.out) if args.out else config.ROOT_DIR / "results" / "gemini_box2d" / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    polygons, confs = yolo_polygons(img)
    if not polygons:
        raise SystemExit("YOLO 沒有偵測到紙箱")
    boxes = [to_box_2d(p, w, h) for p in polygons]
    print(f"圖片：{image_path}  紙箱：{len(polygons)} 個 → 輸出到 {out_dir}")

    image_jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
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
