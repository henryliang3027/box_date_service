"""
test_api_crop_box_gemini.py — 「YOLO 裁切逐箱呼叫 Gemini」方案效能測試腳本
================================================================
與 service_api/pipeline.py 目前採用的「整圖丟給 Gemini，再用中心點比對 YOLO mask」
不同，此腳本模擬另一種作法：

    YOLO segmentation → 取得每個紙箱的 mask
      → 依 mask 的 bounding box 裁切出單一紙箱小圖
      → 對每張小圖各自呼叫一次 Gemini API 辨識
      → 逐箱時間加總

用來實測這種「一張圖 N 個箱子 = N 次 Gemini API call」的作法在 latency 上
是否真的比整圖一次 Gemini call（見 test_api_detection_result.py 量到的數據）高很多。

直接在本機呼叫 YOLO / Gemini（不透過 FastAPI），純粹測兩段的耗時：
  - yolo_time_s      : YOLO segmentation 耗時（一張圖一次）
  - gemini_time_s    : 該圖所有紙箱裁切後，逐一呼叫 Gemini 的總耗時
  - gemini_calls     : 該圖呼叫 Gemini 的次數（= 偵測到的紙箱數）
  - total_time_s     : yolo_time_s + gemini_time_s

耗時資料寫入 logs/timing_log_crop.csv，執行前會先清空。

使用方式：
    python3 test_api_crop_box_gemini.py [圖片資料夾路徑]

範例：
    python3 test_api_crop_box_gemini.py
    python3 test_api_crop_box_gemini.py /home/ubuntu/Documents/API_Test/test_images/v1_release
"""

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from service_api.services.detector import BoxDetector
from service_api.services.gemini_client import GeminiBoxClient

DEFAULT_IMAGE_DIR = "/home/ubuntu/Documents/API_Test/test_images/博亞藥局/"

REPO_ROOT       = Path(__file__).resolve().parent
TIMING_LOG_PATH = REPO_ROOT / "logs" / "timing_log_crop.csv"
IMAGE_SUFFIXES  = (".jpg", ".jpeg", ".png")

# 裁切時在 mask bounding box 四周額外保留的邊界（像素），避免文字被切邊
CROP_PADDING = 20


def crop_boxes(img_bgr: np.ndarray, polygons: list[np.ndarray]) -> list[bytes]:
    """依每個 mask polygon 的 bounding box，從原圖裁切出單一紙箱小圖，回傳 jpg bytes 列表。"""
    img_h, img_w = img_bgr.shape[:2]
    crops: list[bytes] = []
    for polygon in polygons:
        x, y, w, h = cv2.boundingRect(polygon.astype(np.float32))
        x0 = max(0, int(x) - CROP_PADDING)
        y0 = max(0, int(y) - CROP_PADDING)
        x1 = min(img_w, int(x + w) + CROP_PADDING)
        y1 = min(img_h, int(y + h) + CROP_PADDING)

        crop = img_bgr[y0:y1, x0:x1]
        ok, buf = cv2.imencode(".jpg", crop)
        if not ok:
            continue
        crops.append(buf.tobytes())
    return crops


def process_image(
    img_path: Path,
    detector: BoxDetector,
    gemini: GeminiBoxClient,
) -> dict:
    """對單張圖片執行「YOLO 裁切逐箱呼叫 Gemini」流程，回傳耗時統計。"""
    img_bgr = cv2.imread(str(img_path))

    yolo_t0   = time.time()
    detection = detector.detect(img_bgr)
    yolo_time = time.time() - yolo_t0

    crops = crop_boxes(img_bgr, detection.polygons)

    gemini_time = 0.0
    for i, crop_bytes in enumerate(crops):
        gemini_t0 = time.time()
        gemini.detect(crop_bytes, filename=f"{img_path.stem}_box{i}.jpg")
        gemini_time += time.time() - gemini_t0

    return {
        "filename":       img_path.name,
        "num_boxes":      len(crops),
        "yolo_time_s":    yolo_time,
        "gemini_time_s":  gemini_time,
        "total_time_s":   yolo_time + gemini_time,
    }


def run(image_path: Path) -> None:
    if image_path.is_file():
        images = [image_path]
    else:
        images = sorted(
            p for p in image_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
    if not images:
        print(f"在 {image_path} 找不到任何圖片。")
        return

    detector = BoxDetector()
    gemini   = GeminiBoxClient()

    TIMING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TIMING_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "num_boxes", "yolo_time_s", "gemini_time_s", "total_time_s"])

        for img_path in images:
            print(f"Processing {img_path.name} ...")
            row = process_image(img_path, detector, gemini)
            print(
                f"  boxes={row['num_boxes']} "
                f"yolo={row['yolo_time_s']:.3f}s "
                f"gemini={row['gemini_time_s']:.3f}s "
                f"total={row['total_time_s']:.3f}s"
            )
            writer.writerow([
                row["filename"],
                row["num_boxes"],
                f"{row['yolo_time_s']:.3f}",
                f"{row['gemini_time_s']:.3f}",
                f"{row['total_time_s']:.3f}",
            ])


def print_timing_table() -> None:
    """讀取 logs/timing_log_crop.csv，印出整理好的表格。"""
    if not TIMING_LOG_PATH.exists():
        print(f"找不到 timing log：{TIMING_LOG_PATH}")
        return

    with open(TIMING_LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("timing log 是空的。")
        return

    box_sum    = sum(int(r["num_boxes"]) for r in rows)
    yolo_sum   = sum(float(r["yolo_time_s"]) for r in rows)
    gemini_sum = sum(float(r["gemini_time_s"]) for r in rows)
    total_sum  = sum(float(r["total_time_s"]) for r in rows)
    n = len(rows)

    header = (
        f"{'圖片':<16}{'箱數':<6}{'YOLO segmentation (s)':<24}"
        f"{'Gemini inference (s)':<24}{'總花費時間 (s)':<16}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['filename']:<16}{r['num_boxes']:<6}{r['yolo_time_s']:<24}"
            f"{r['gemini_time_s']:<24}{r['total_time_s']:<16}"
        )
    print("-" * len(header))
    print(
        f"{'平均':<16}{box_sum/n:<6.1f}{yolo_sum/n:<24.3f}"
        f"{gemini_sum/n:<24.3f}{total_sum/n:<16.3f}"
    )


def main() -> None:
    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_IMAGE_DIR)
    run(image_dir)
    print_timing_table()


if __name__ == "__main__":
    main()
