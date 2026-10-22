"""
test_gemini_batch.py — 「依 YOLO polygon 切出每個紙箱 + Gemini Batch API」辨識品名的實驗腳本
=============================================================================================
  1. YOLO segmentation（服務的模型與參數）取得每個紙箱的 polygon（result.masks.xy，原圖像素座標）
  2. 依 polygon 切出每個紙箱：取 polygon 的外接矩形，矩形內 polygon 以外的像素塗成白色
  3. 每個紙箱一個 request，用 Gemini Batch API（inline requests）一次送出，輪詢到完成
  4. 依 id 把結果和 polygon 合併，輸出 result.json：[{id, name, manufacturer_date, expiration_date, conf, polygon}]
prompt 沿用服務（gemini_client.py）的規則，去掉 box_2d。

注意：Batch API 是非同步的（官方目標 24 小時內完成，費用為一般 API 的 50%），
排隊時間不固定，適合離線大量處理，不適合 /detect 這種要即時回應的流程。

使用方式：
    python3 test_gemini_batch.py --image sample_images/resized.jpg [--out 輸出資料夾]

輸出（預設 results/gemini_batch/<圖片檔名>/）：
    crop_<id>.jpg  送給 Gemini 的紙箱切圖
    result.json    [{id, name, manufacturer_date, expiration_date, conf, polygon}]，polygon 為原圖像素座標 [[x, y], ...]
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
from service_api.services.gemini_client import _DATE_SCHEMA, PROMPT as SERVICE_PROMPT

# 辨識規則沿用服務的 prompt 第 1～3 條（日期、品名、紙箱偵測），去掉第 4 條 box_2d：
# 紙箱位置已由 YOLO 給定，不需要 Gemini 定位，確保實驗只差在「怎麼指定紙箱」
_RULES = SERVICE_PROMPT[SERVICE_PROMPT.index("1. 日期"):SERVICE_PROMPT.index("4. box_2d")].strip()

PROMPT = f"""你是專業的商品紙箱文字辨識專家。這張圖是從照片中依輪廓切下來的「單一個」紙箱，
紙箱輪廓以外的區域已塗成白色，不屬於這個紙箱。請分析這個紙箱，並遵循以下規則：

{_RULES}

4. 紙箱已切好，固定只回傳這一個紙箱的一筆物件：只看這個紙箱本身的箱面文字，切圖邊緣可能殘留相鄰紙箱的一小部分，不可採用；
   依第 3 條應忽略的情況（被遮擋大半、只露極小一角）仍要回傳，product_name 填 "未知"、日期填 null，不可猜測。

輸出這個紙箱的結構化物件。"""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "product_name": {"type": "STRING"},
        "manufacturer_date": _DATE_SCHEMA,
        "expiration_date": _DATE_SCHEMA,
    },
    "required": ["product_name", "manufacturer_date", "expiration_date"],
}

_DONE_STATES = {
    "JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED",
    "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
}


def yolo_polygons(img: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """用服務的 YOLO 設定跑 segmentation，回傳每個紙箱的 polygon（原圖像素座標）與信心值。"""
    from ultralytics import YOLO
    r = YOLO(config.YOLO_MODEL_PATH).predict(
        img, task="seg", imgsz=config.YOLO_IMG_SIZE, conf=config.YOLO_CONF_THRESHOLD, verbose=False,
    )[0]
    if r.masks is None:
        return [], []
    # 少於 3 點的退化輪廓切不出區域，略過
    pairs = [(p, c) for p, c in zip(r.masks.xy, r.boxes.conf.tolist()) if len(p) >= 3]
    return [p for p, _ in pairs], [c for _, c in pairs]


def crop_by_polygon(img: np.ndarray, poly: np.ndarray, pad: int = 8) -> np.ndarray:
    """取 polygon 外接矩形（外擴 pad 像素），矩形內 polygon 以外的像素塗白。"""
    h, w = img.shape[:2]
    pts = np.round(poly).astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(w, x + bw + pad), min(h, y + bh + pad)
    mask = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillPoly(mask, [pts - [x1, y1]], 1)
    crop = img[y1:y2, x1:x2].copy()
    crop[mask == 0] = 255
    return crop


def run_batch(client: genai.Client, crops: list[np.ndarray], poll_s: float, timeout_s: float) -> dict[int, dict]:
    """每個切圖一個 inline request，建立 batch job、輪詢到完成，回傳 {id: {product_name, manufacturer_date, expiration_date}}。"""
    requests = []
    for i, crop in enumerate(crops):
        jpg = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        requests.append(types.InlinedRequest(
            contents=[types.Content(role="user", parts=[
                types.Part.from_bytes(data=jpg, mime_type="image/jpeg"),
                types.Part.from_text(text=PROMPT),
            ])],
            metadata={"id": str(i)},
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(thinking_level=types.ThinkingLevel.HIGH),
                temperature=0,
                response_mime_type="application/json",
                response_schema=SCHEMA,
            ),
        ))

    job = client.batches.create(
        model=config.GEMINI_MODEL,
        src=requests,
        config=types.CreateBatchJobConfig(display_name=f"box-names-{int(time.time())}"),
    )
    print(f"[BATCH] 建立 {job.name}（{len(requests)} 個 request）")

    t0 = time.perf_counter()
    while True:
        job = client.batches.get(name=job.name)
        state = job.state.name
        elapsed = time.perf_counter() - t0
        print(f"[BATCH] {elapsed:6.0f}s  {state}")
        if state in _DONE_STATES:
            break
        if elapsed > timeout_s:
            raise SystemExit(f"等待超過 {timeout_s:.0f}s 仍未完成，job 保留在伺服器上：{job.name}"
                             f"（之後可用 client.batches.get(name=...) 取回結果）")
        time.sleep(poll_s)

    if state not in ("JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"):
        raise SystemExit(f"batch job 失敗：{state} {job.error}")

    names: dict[int, dict] = {}
    for idx, r in enumerate(job.dest.inlined_responses):
        # 有 metadata 就以 metadata 的 id 為準，否則依送出順序
        i = int(r.metadata["id"]) if r.metadata and "id" in r.metadata else idx
        if r.error:
            print(f"⚠️ #{i} 失敗：{r.error}")
            continue
        names[i] = json.loads(r.response.text)
        usage = r.response.usage_metadata
        print(f"[BATCH] #{i} tokens: prompt={usage.prompt_token_count} "
              f"thoughts={usage.thoughts_token_count or 0} output={usage.candidates_token_count or 0}")
    return names


def fmt_date(d: dict | None) -> str:
    return f"{d['year']}-{d['month']}-{d['day']}" if d else "-"


def main() -> None:
    parser = argparse.ArgumentParser(description="依 YOLO polygon 切出紙箱 → Gemini Batch API 辨識品名")
    parser.add_argument("--image",   required=True, help="紙箱圖片")
    parser.add_argument("--out",     help="輸出資料夾（預設 results/gemini_batch/<圖片檔名>）")
    parser.add_argument("--poll",    type=float, default=10,   help="輪詢間隔秒數")
    parser.add_argument("--timeout", type=float, default=1800, help="最多等幾秒")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"無法讀取圖片：{image_path}")
    out_dir = Path(args.out) if args.out else config.ROOT_DIR / "results" / "gemini_batch" / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    polygons, confs = yolo_polygons(img)
    if not polygons:
        raise SystemExit("YOLO 沒有偵測到紙箱")
    print(f"圖片：{image_path}  紙箱：{len(polygons)} 個 → 輸出到 {out_dir}")

    crops = [crop_by_polygon(img, p) for p in polygons]
    for i, c in enumerate(crops):
        cv2.imwrite(str(out_dir / f"crop_{i}.jpg"), c)

    names = run_batch(genai.Client(api_key=config.GEMINI_API_KEY), crops, args.poll, args.timeout)

    missing = [i for i in range(len(polygons)) if i not in names]
    if missing:
        print(f"⚠️ 沒有取得這些編號的結果：{missing}")

    result = [
        {
            "id": i,
            "name": names.get(i, {}).get("product_name"),
            "manufacturer_date": names.get(i, {}).get("manufacturer_date"),
            "expiration_date": names.get(i, {}).get("expiration_date"),
            "conf": round(c, 3),
            "polygon": np.round(p, 1).tolist(),
        }
        for i, (p, c) in enumerate(zip(polygons, confs))
    ]
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")

    for r, c in zip(result, crops):
        print(f"  #{r['id']}  conf={r['conf']:.3f}  crop={c.shape[1]}x{c.shape[0]}  name={r['name']}  "
              f"mfg={fmt_date(r['manufacturer_date'])}  exp={fmt_date(r['expiration_date'])}")
    print(f"→ {out_dir / 'result.json'}")


if __name__ == "__main__":
    main()
