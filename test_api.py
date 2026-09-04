"""
test_api.py — /api/v1/detect/image 效能測試腳本
==================================================
對指定資料夾內所有圖片依序呼叫 /api/v1/detect/image，
並統計每張圖片的 YOLO segmentation / Gemini inference 花費時間。

耗時資料來源：service_api/pipeline.py 的 _log_timing()，
每次請求都會把耗時附加寫入 logs/timing_log.csv，
本腳本執行前會先清空該檔案，執行後讀取並整理成表格。

使用方式：
    python3 test_api.py [圖片資料夾路徑] [API base URL]

範例：
    python3 test_api.py
    python3 test_api.py /home/ubuntu/Documents/API_Test/test_images/v1_release
    python3 test_api.py /path/to/images http://localhost:8080
"""

import csv
import sys
from pathlib import Path

import requests

DEFAULT_IMAGE_DIR = "/home/ubuntu/Documents/API_Test/test_images/v2"
DEFAULT_API_BASE  = "http://localhost:8080"

REPO_ROOT       = Path(__file__).resolve().parent
TIMING_LOG_PATH = REPO_ROOT / "logs" / "timing_log.csv"
IMAGE_SUFFIXES  = (".jpg", ".jpeg", ".png")


def run_detect(image_dir: Path, api_base: str) -> None:
    """對資料夾內所有圖片依序呼叫 /api/v1/detect/image。"""
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        print(f"在 {image_dir} 找不到任何圖片。")
        return

    url = f"{api_base}/api/v1/detect/image"
    for img_path in images:
        print(f"Processing {img_path.name} ...")
        with open(img_path, "rb") as f:
            resp = requests.post(
                url,
                files={"file": f},
                data={"include_annotated_image": "false"},
            )
        resp.raise_for_status()


def print_timing_table() -> None:
    """讀取 logs/timing_log.csv，印出整理好的表格。"""
    if not TIMING_LOG_PATH.exists():
        print(f"找不到 timing log：{TIMING_LOG_PATH}")
        return

    with open(TIMING_LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("timing log 是空的。")
        return

    yolo_sum   = sum(float(r["yolo_time_s"])   for r in rows)
    gemini_sum = sum(float(r["gemini_time_s"]) for r in rows)
    total_sum  = sum(float(r["total_time_s"])  for r in rows)
    n = len(rows)

    header = f"{'圖片':<16}{'YOLO segmentation (s)':<24}{'Gemini inference (s)':<24}{'總花費時間 (s)':<16}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['filename']:<16}{r['yolo_time_s']:<24}{r['gemini_time_s']:<24}{r['total_time_s']:<16}")
    print("-" * len(header))
    print(f"{'平均':<16}{yolo_sum/n:<24.3f}{gemini_sum/n:<24.3f}{total_sum/n:<16.3f}")


def main() -> None:
    image_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_IMAGE_DIR)
    api_base  = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_API_BASE

    # 先清空舊的 timing log，避免混到上次測試的資料
    TIMING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    TIMING_LOG_PATH.unlink(missing_ok=True)

    run_detect(image_dir, api_base)
    print_timing_table()


if __name__ == "__main__":
    main()
