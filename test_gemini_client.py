"""
test_gemini_client.py — 服務 GeminiBoxClient（整張圖推論）的測試腳本
====================================================================
只跑服務的 service_api/services/gemini_client.py（不跑 YOLO），檢查 Gemini 自己
偵測紙箱 + 辨識品名/日期的表現：
  1. 原圖 bytes 直接傳給 GeminiBoxClient.detect（與服務 pipeline 相同，不重新編碼）
  2. 回傳的 box_2d（[ymin, xmin, ymax, xmax]，0-1000）還原成像素矩形
  3. 畫出矩形 + 中心點「#id / 品名 / 日期」標籤 + 紅色中心點，並輸出 summary.json

使用方式：
    python3 test_gemini_client.py --source <圖片或資料夾> [--out 輸出資料夾]

範例：
    python3 test_gemini_client.py --source sample_images/全家
    python3 test_gemini_client.py --source "sample_images/全家/正面-2.jpg"

輸出（預設 results/gemini_client/<來源資料夾名稱>/）：
    <圖片檔名>_gemini.jpg  原圖 + Gemini box_2d 矩形 + 品名/日期標籤 + 紅色中心點
    summary.json           每張圖的紙箱數、耗時、Gemini 原始回傳
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from service_api import config
from service_api.services.gemini_client import GeminiBoxClient
from test_gemini_contour import PALETTE, draw_result
from test_yolo26_seg import list_images


def box_2d_to_result(i: int, box: dict, w: int, h: int) -> dict:
    """Gemini 回傳的一筆紙箱 → draw_result 需要的格式（box_2d 矩形當作 polygon）。"""
    ymin, xmin, ymax, xmax = box["box_2d"]
    x1, y1, x2, y2 = xmin / 1000 * w, ymin / 1000 * h, xmax / 1000 * w, ymax / 1000 * h
    return {
        "id": i,
        "color": PALETTE[i % len(PALETTE)][0],
        "name": box.get("product_name"),
        "manufacturer_date": box.get("manufacturer_date"),
        "expiration_date": box.get("expiration_date"),
        "center": [(x1 + x2) / 2, (y1 + y2) / 2],
        "polygon": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    }


def draw_centers(pil: Image.Image, result: list[dict]) -> Image.Image:
    """在每個 box_2d 中心點畫紅色實心圓點（白色外框，壓在標籤上也看得到）。"""
    r = max(4, round(max(pil.size) / 160))       # 圓點半徑依圖片大小縮放
    draw = ImageDraw.Draw(pil)
    for item in result:
        cx, cy = item["center"]
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 0, 0), outline=(255, 255, 255), width=2)
    return pil


def main() -> None:
    parser = argparse.ArgumentParser(description="測試服務的 GeminiBoxClient（整張圖推論）")
    parser.add_argument("--source", required=True, help="圖片檔或資料夾")
    parser.add_argument("--out",    help="輸出資料夾（預設 results/gemini_client/<來源資料夾名稱>）")
    args = parser.parse_args()

    source = Path(args.source)
    images = list_images(source)
    if not images:
        raise SystemExit(f"找不到圖片：{source}")

    folder_name = source.name if source.is_dir() else source.parent.name
    out_dir = Path(args.out) if args.out else config.ROOT_DIR / "results" / "gemini_client" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    client = GeminiBoxClient()
    print(f"模型：{config.GEMINI_MODEL}")
    print(f"圖片：{len(images)} 張 → 輸出到 {out_dir}\n")

    summary = []
    for path in images:
        image_bytes = path.read_bytes()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # 同服務 pipeline
        if img is None:
            print(f"⚠️ 無法讀取 {path.name}，略過")
            continue
        h, w = img.shape[:2]

        print(f"── {path.name}")
        t0 = time.perf_counter()
        try:
            boxes = client.detect(image_bytes, filename=path.name)
        except Exception as e:
            print(f"⚠️ Gemini 呼叫失敗：{e}\n")
            summary.append({"image": path.name, "width": w, "height": h, "error": str(e)})
            continue
        sec = time.perf_counter() - t0

        result = [box_2d_to_result(i, b, w, h) for i, b in enumerate(boxes)]
        out_path = out_dir / f"{path.stem}_gemini.jpg"
        draw_centers(draw_result(img, result), result).save(out_path, quality=95)
        print(f"{path.name:<20} {w}x{h:<6} 紙箱 {len(boxes):>2}   {sec:6.2f} s\n")

        summary.append({
            "image": path.name,
            "width": w,
            "height": h,
            "num_boxes": len(boxes),
            "gemini_sec": round(sec, 2),
            "boxes": [{"id": i, **b} for i, b in enumerate(boxes)],
        })

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "model": config.GEMINI_MODEL, "images": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [s for s in summary if "error" not in s]
    total = sum(s["num_boxes"] for s in ok)
    avg_sec = sum(s["gemini_sec"] for s in ok) / max(1, len(ok))
    print(f"共 {len(ok)}/{len(summary)} 張成功、{total} 個紙箱，Gemini 平均 {avg_sec:.2f} s")
    print(f"標注圖與 summary.json 已存到 {out_dir}")


if __name__ == "__main__":
    main()
