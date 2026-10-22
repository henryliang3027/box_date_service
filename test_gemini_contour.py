"""
test_gemini_contour.py — 「YOLO 輪廓用不同色號標示 + 整張圖送 Gemini」辨識品名的實驗腳本
==========================================================================================
  1. YOLO segmentation（服務的模型與參數）取得每個紙箱的 polygon（result.masks.xy，原圖像素座標）
  2. 每個紙箱分配一個色號（PALETTE），並由 polygon 算出紙箱中心點（面積重心）；
     色號 → {紙箱 id, 中心點, polygon} 的對應先記錄在 color_map.json
  3. 沿紙箱輪廓「內側」畫一圈該色號的線（不填色、不標數字），相鄰紙箱的線不會重疊
  4. 只傳這張畫好輪廓的圖給 Gemini，prompt 列出圖上用到的色號，請它依色號回品名與日期
     （prompt 沿用服務 gemini_client.py 的規則，去掉 box_2d；回傳的 color 用 enum 限定只能是用到的色號）
  5. 依色號查 color_map 對回紙箱，輸出 result.json，並把品名寫在各紙箱的中心點畫成 result.jpg

使用方式：
    python3 test_gemini_contour.py --image "sample_images/全家/正面-2.jpg" [--out 輸出資料夾]

輸出（預設 results/gemini_contour/<圖片檔名>/）：
    contour.jpg     傳給 Gemini 的輪廓圖
    color_map.json  {色號: {id, color_name, center, conf, polygon}}，center 為 [x, y] 原圖像素座標
    result.json     [{id, color, color_name, name, manufacturer_date, expiration_date, center, conf, polygon}]
    result.jpg      原圖 + YOLO polygon 輪廓（顏色 = 色號）+ 每個紙箱中心點寫上品名（標籤底色 = 色號），左上角為總數與品項統計
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from collections import Counter
from google import genai
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

from service_api import config
from service_api.services.gemini_client import _DATE_SCHEMA, PROMPT as SERVICE_PROMPT

# 辨識規則沿用服務的 prompt 第 1～3 條（日期、品名、紙箱偵測），去掉第 4 條 box_2d
_RULES = SERVICE_PROMPT[SERVICE_PROMPT.index("1. 日期"):SERVICE_PROMPT.index("4. box_2d")].strip()

# 色號（RGB hex）與中文名稱；越前面越容易彼此區分，紙箱數超過色號數量時無法標示
PALETTE: list[tuple[str, str]] = [
    ("#FF0000", "紅"),   ("#0000FF", "藍"),   ("#00FF00", "綠"),   ("#FFFF00", "黃"),
    ("#FF00FF", "洋紅"), ("#00FFFF", "青"),   ("#FF8000", "橘"),   ("#8000FF", "紫"),
    ("#000000", "黑"),   ("#FFFFFF", "白"),   ("#FF80C0", "粉紅"), ("#000080", "深藍"),
    ("#804000", "棕"),   ("#808000", "橄欖"), ("#008080", "藍綠"), ("#808080", "灰"),
    ("#00A0FF", "天藍"), ("#800000", "栗"),   ("#C0FF00", "萊姆"), ("#C080FF", "薰衣草"),
]

PROMPT = """你是專業的商品紙箱文字辨識專家。圖中每個紙箱都沿著輪廓內側畫了一圈彩色線（只有線，內部未填色），
每個紙箱的線條顏色都不同，作為該紙箱的識別色號。彩色線是後製加上的標記，不屬於箱面文字，辨識時請忽略。

圖上用到的色號（共 {n} 個紙箱）：
{color_list}

請依色號逐一分析「每一個」紙箱線框內的資訊，每個紙箱各自獨立為清單中一個物件，並遵循以下規則：

{rules}

4. 紙箱已由圖上的彩色線框指定：上面列出的每個色號都要回傳一筆，color 填該紙箱線框的色號（如 "#FF0000"），
   不可回傳清單以外的色號、不可合併或拆分；判斷顏色時以線條本身的顏色為準，不要和箱面印刷的顏色混淆。
   依第 3 條應忽略的紙箱（被遮擋大半、只露極小一角）仍要回傳一筆，product_name 填 "未知"、日期填 null，不可從相鄰紙箱推測。

輸出所有色號紙箱的結構化物件陣列。"""


def hex_to_bgr(h: str) -> tuple[int, int, int]:
    r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
    return b, g, r


def yolo_polygons(img: np.ndarray) -> tuple[list[np.ndarray], list[float]]:
    """用服務的 YOLO 設定跑 segmentation，回傳每個紙箱的 polygon（原圖像素座標）與信心值。"""
    from ultralytics import YOLO
    r = YOLO(config.YOLO_MODEL_PATH).predict(
        img, task="seg", imgsz=config.YOLO_IMG_SIZE, conf=config.YOLO_CONF_THRESHOLD, verbose=False,
    )[0]
    if r.masks is None:
        return [], []
    # 少於 3 點的退化輪廓畫不出區域，略過
    pairs = [(p, c) for p, c in zip(r.masks.xy, r.boxes.conf.tolist()) if len(p) >= 3]
    return [p for p, _ in pairs], [c for _, c in pairs]


def polygon_center(poly: np.ndarray) -> tuple[float, float]:
    """polygon 的面積重心；輪廓退化（面積為 0）時改用外接矩形中心。"""
    m = cv2.moments(poly.astype(np.float32))
    if m["m00"] > 0:
        return m["m10"] / m["m00"], m["m01"] / m["m00"]
    x, y, w, h = cv2.boundingRect(np.round(poly).astype(np.int32))
    return x + w / 2, y + h / 2


def draw_contours(img: np.ndarray, polygons: list[np.ndarray], colors: list[str]) -> np.ndarray:
    """
    每個紙箱沿輪廓內側畫一圈色帶：mask 減掉侵蝕後的 mask = 內側寬度 thick 的環。
    畫在內側而不是輪廓線上，相鄰紙箱共用的邊界才不會被後畫的顏色蓋掉。
    """
    h, w = img.shape[:2]
    thick = max(4, round(5 * max(w, h) / 1000))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * thick + 1, 2 * thick + 1))

    out = img.copy()
    for poly, color in zip(polygons, colors):
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
        ring = mask & (1 - cv2.erode(mask, kernel))
        out[ring == 1] = hex_to_bgr(color)
    return out


def ask_gemini(contour_jpg: bytes, colors: list[str]) -> list[dict]:
    names = dict(PALETTE)
    color_list = "\n".join(f"- {c}（{names[c]}）" for c in colors)
    schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "color": {"type": "STRING", "enum": colors, "description": "紙箱線框的色號"},
                "product_name": {"type": "STRING"},
                "manufacturer_date": _DATE_SCHEMA,
                "expiration_date": _DATE_SCHEMA,
            },
            "required": ["color", "product_name", "manufacturer_date", "expiration_date"],
        },
    }

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=contour_jpg, mime_type="image/jpeg"),
            PROMPT.format(n=len(colors), color_list=color_list, rules=_RULES),
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


def fmt_date(d: dict | None) -> str:
    return f"{d['year']}-{d['month']}-{d['day']}" if d else "-"


def draw_result(img: np.ndarray, result: list[dict]) -> Image.Image:
    """
    在原圖上畫出每個紙箱的 YOLO polygon 輪廓（顏色 = 該紙箱色號），
    並以紙箱中心點為中心寫上「#id 色名 / 品名 / 日期」，標籤底色 = 該紙箱色號。
    """
    h, w = img.shape[:2]
    thick = max(2, round(3 * max(w, h) / 1000))
    out = img.copy()
    for r in result:
        cv2.polylines(out, [np.round(np.array(r["polygon"])).astype(np.int32)], True, hex_to_bgr(r["color"]), thick)

    font = ImageFont.truetype(config.FONT_PATH, max(14, round(w / 50)))
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)

    for r in result:
        rgb = tuple(int(r["color"][i:i + 2], 16) for i in (1, 3, 5))
        dark_bg = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2] <= 140
        text_color, border = ((255, 255, 255), (0, 0, 0)) if dark_bg else ((0, 0, 0), (255, 255, 255))

        label = f"#{r['id']} {r.get('color_name', '')}".rstrip() + f"\n{r['name'] or '（Gemini 未回）'}"
        for key, tag in (("manufacturer_date", "製"), ("expiration_date", "效")):
            if r[key]:
                label += f"\n{tag}: {fmt_date(r[key])}"

        # 標籤以紙箱中心點為中心，超出圖片時往內貼齊
        cx, cy = r["center"]
        l, t, rr, b = draw.multiline_textbbox((0, 0), label, font=font)
        tw, th = rr - l + 8, b - t + 8
        x1 = int(min(max(0, cx - tw / 2), w - tw))
        y1 = int(min(max(0, cy - th / 2), h - th))
        draw.rectangle([x1, y1, x1 + tw, y1 + th], fill=rgb, outline=border, width=2)
        draw.multiline_text((x1 + 4 - l, y1 + 4 - t), label, fill=text_color, font=font)

    # 左上角：總數 + 各品項數量
    counts = Counter(r["name"] for r in result)
    summary = "\n".join([f"總數量: {len(result)}"] + [f"{n}: {c}" for n, c in counts.most_common()])
    l, t, rr, b = draw.multiline_textbbox((0, 0), summary, font=font)
    draw.rectangle([0, 0, rr - l + 8, b - t + 8], fill=(0, 0, 0), outline=(255, 255, 255), width=2)
    draw.multiline_text((4 - l, 4 - t), summary, fill=(255, 255, 255), font=font)
    return pil


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO 輪廓用不同色號標示 → Gemini 依色號辨識品名")
    parser.add_argument("--image", required=True, help="紙箱圖片")
    parser.add_argument("--out",   help="輸出資料夾（預設 results/gemini_contour/<圖片檔名>）")
    args = parser.parse_args()

    image_path = Path(args.image)
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit(f"無法讀取圖片：{image_path}")
    out_dir = Path(args.out) if args.out else config.ROOT_DIR / "results" / "gemini_contour" / image_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    polygons, confs = yolo_polygons(img)
    if not polygons:
        raise SystemExit("YOLO 沒有偵測到紙箱")
    if len(polygons) > len(PALETTE):
        raise SystemExit(f"紙箱 {len(polygons)} 個，超過色號數量 {len(PALETTE)}，無法一箱一色")
    colors = [c for c, _ in PALETTE[:len(polygons)]]
    color_names = dict(PALETTE)
    print(f"圖片：{image_path}  紙箱：{len(polygons)} 個 → 輸出到 {out_dir}")

    # 先記下色號 ↔ 紙箱（id、中心點、polygon）的對應表，Gemini 回傳的色號靠它對回紙箱
    color_map = {
        c: {
            "id": i,
            "color_name": color_names[c],
            "center": [round(v, 1) for v in polygon_center(p)],
            "conf": round(conf, 3),
            "polygon": np.round(p, 1).tolist(),
        }
        for i, (c, p, conf) in enumerate(zip(colors, polygons, confs))
    }
    (out_dir / "color_map.json").write_text(json.dumps(color_map, ensure_ascii=False), encoding="utf-8")

    contour = draw_contours(img, polygons, colors)
    cv2.imwrite(str(out_dir / "contour.jpg"), contour)
    contour_jpg = cv2.imencode(".jpg", contour, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()

    t0 = time.perf_counter()
    answers = ask_gemini(contour_jpg, colors)
    print(f"[GEMINI] {time.perf_counter() - t0:.2f}s")

    # 依色號合併；Gemini 漏回的色號 name 為 None，重複的色號只採用第一筆
    by_color: dict[str, dict] = {}
    for a in answers:
        c = a["color"].upper()
        if c in color_map and c not in by_color:
            by_color[c] = a
        else:
            print(f"⚠️ 忽略 Gemini 回傳的無效或重複色號：{a}")
    missing = [f"{c}（{color_names[c]}）" for c in colors if c not in by_color]
    if missing:
        print(f"⚠️ Gemini 沒回這些色號：{missing}")

    result = []
    for c in colors:
        a, m = by_color.get(c, {}), color_map[c]
        result.append({
            "id": m["id"],
            "color": c,
            "color_name": m["color_name"],
            "name": a.get("product_name"),
            "manufacturer_date": a.get("manufacturer_date"),
            "expiration_date": a.get("expiration_date"),
            "center": m["center"],
            "conf": m["conf"],
            "polygon": m["polygon"],
        })
    (out_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    draw_result(img, result).save(out_dir / "result.jpg", quality=95)

    for r in result:
        print(f"  #{r['id']:<2} {r['color']} {r['color_name']:<3}  center=({r['center'][0]:.0f}, {r['center'][1]:.0f})  name={r['name']}  "
              f"mfg={fmt_date(r['manufacturer_date'])}  exp={fmt_date(r['expiration_date'])}")
    print(f"→ {out_dir / 'result.json'}、{out_dir / 'result.jpg'}")


if __name__ == "__main__":
    main()
