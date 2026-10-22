"""
test_gemini_rle.py — 「原圖 + 每個紙箱的壓縮 RLE」讓 Gemini 依 id 辨識品名的實驗腳本
======================================================================================
圖上不畫任何標記，直接把每個紙箱 mask 的 COCO 壓縮 RLE（pycocotools）以文字傳給 Gemini，
測試 Gemini 能不能從 RLE 字串看出紙箱在圖上的位置：
  1. YOLO segmentation（服務的模型與參數，retina_masks=True）取得每個紙箱原圖尺寸的二值 mask 與 polygon
  2. mask 用 pycocotools 編成壓縮 RLE：{"size": [H, W], "counts": "<字串>"}
  3. 原圖 + 「id → 壓縮 RLE」清單一起傳給 Gemini，請它依 id 回品名與日期
     （prompt 沿用服務的規則，去掉 box_2d 偵測；回傳的 id 用 enum 限定）
  4. 依 id 對回紙箱，輸出 result.json，並畫出 YOLO 輪廓 + 中心點品名的 result.jpg

使用方式：
    python3 test_gemini_rle.py --image "sample_images/全家/正面-2.jpg" [--out 輸出資料夾]

輸出（預設 results/gemini_rle/<圖片檔名>/）：
    result.json  [{id, rle, name, manufacturer_date, expiration_date, center, conf, polygon}]
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
from pycocotools import mask as mask_util

from service_api import config
from service_api.services.gemini_client import _DATE_SCHEMA
from test_gemini_contour import PALETTE, _RULES, draw_result, fmt_date, polygon_center

PROMPT = """你是專業的商品紙箱文字辨識專家。圖中紙箱的位置已經偵測好，共 {n} 個，
每個紙箱以 COCO 壓縮 RLE（pycocotools 格式）表示其 segmentation mask：
- size 為 [圖高, 圖寬]（單位：像素）
- counts 為壓縮後的字串：mask 依 column-major（由左而右、每一行由上而下）展開，
  記錄 0、1 交替的連續長度（第一段為 0 的長度），再做差分與每 5 bit 一組的 ASCII 可變長度編碼

{rle_list}

請依 id 逐一分析「每一個」紙箱 mask 範圍內的資訊，每個紙箱各自獨立為清單中一個物件，並遵循以下規則：

{rules}

4. 紙箱已由上面的 id 與 RLE 指定：每個 id 都要回傳一筆，不可新增清單以外的 id、不可合併或拆分；
   依第 3 條應忽略的紙箱（被遮擋大半、只露極小一角）仍要回傳一筆，product_name 填 "未知"、日期填 null，不可從相鄰紙箱推測。

輸出所有 id 紙箱的結構化物件陣列。"""


def yolo_masks(img: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray], list[float]]:
    """YOLO segmentation（retina_masks=True），回傳原圖尺寸的二值 mask、polygon（原圖像素座標）與信心值。"""
    from ultralytics import YOLO
    r = YOLO(config.YOLO_MODEL_PATH).predict(
        img, task="seg", imgsz=config.YOLO_IMG_SIZE, conf=config.YOLO_CONF_THRESHOLD,
        retina_masks=True, verbose=False,
    )[0]
    if r.masks is None:
        return [], [], []
    masks = r.masks.data.cpu().numpy().astype(np.uint8)
    # 少於 3 點的退化輪廓畫不出來，連同 mask 一起略過
    keep = [i for i, p in enumerate(r.masks.xy) if len(p) >= 3]
    confs = r.boxes.conf.tolist()
    return [masks[i] for i in keep], [r.masks.xy[i] for i in keep], [confs[i] for i in keep]


def mask_to_coco_rle(m: np.ndarray) -> dict:
    """二值 mask → pycocotools 壓縮 RLE，counts 轉成 str 才能放進 JSON / prompt。"""
    rle = mask_util.encode(np.asfortranarray(m))
    return {"size": [int(v) for v in rle["size"]], "counts": rle["counts"].decode("ascii")}


def ask_gemini(image_jpg: bytes, rles: list[dict]) -> list[dict]:
    rle_list = "\n".join(f"- id {i}: {json.dumps(r)}" for i, r in enumerate(rles))
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                # enum 只接受字串，id 以字串回傳再轉回整數
                "id": {"type": "STRING", "enum": [str(i) for i in range(len(rles))]},
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
            PROMPT.format(n=len(rles), rle_list=rle_list, rules=_RULES),
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
    parser = argparse.ArgumentParser(description="原圖 + 壓縮 RLE → Gemini 依 id 辨識品名")
    parser.add_argument("--image", required=True, help="紙箱圖片")
    parser.add_argument("--out",   help="輸出資料夾（預設 results/gemini_rle/<圖片檔名>）")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"無法讀取圖片：{image_path}")
    out_dir = Path(args.out) if args.out else config.ROOT_DIR / "results" / "gemini_rle" / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    masks, polygons, confs = yolo_masks(img)
    if not masks:
        raise SystemExit("YOLO 沒有偵測到紙箱")
    rles = [mask_to_coco_rle(m) for m in masks]
    total_chars = sum(len(r["counts"]) for r in rles)
    print(f"圖片：{image_path}  紙箱：{len(masks)} 個  RLE counts 共 {total_chars} 字元 → 輸出到 {out_dir}")

    image_jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
    t0 = time.perf_counter()
    answers = ask_gemini(image_jpg, rles)
    print(f"[GEMINI] {time.perf_counter() - t0:.2f}s")

    by_id: dict[int, dict] = {}
    for a in answers:
        i = int(a["id"])
        if i not in by_id:
            by_id[i] = a
        else:
            print(f"⚠️ 忽略 Gemini 回傳的重複 id：{a}")
    missing = [i for i in range(len(masks)) if i not in by_id]
    if missing:
        print(f"⚠️ Gemini 沒回這些 id：{missing}")

    result = []
    for i, (rle, p, c) in enumerate(zip(rles, polygons, confs)):
        a = by_id.get(i, {})
        result.append({
            "id": i,
            "rle": rle,
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
        print(f"  #{r['id']:<2} rle={len(r['rle']['counts']):>4} 字元  name={r['name']}  "
              f"mfg={fmt_date(r['manufacturer_date'])}  exp={fmt_date(r['expiration_date'])}")
    print(f"→ {out_dir / 'result.json'}、{out_dir / 'result.jpg'}")


if __name__ == "__main__":
    main()
