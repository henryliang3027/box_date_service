"""
test_api_pipeline_direct.py — 目前 pipeline（整圖辨識 + mask 比對）效能測試腳本
================================================================
直接在本機呼叫 service_api/pipeline.py 的 BoxDetectionPipeline（不透過 FastAPI），
測量「YOLO segmentation（一次） + Gemini 整圖推論（一次） + 中心點比對」整體流程的耗時。

可與 test_api_crop_box_gemini.py（逐箱裁切呼叫 Gemini 方案）的結果對照比較。

耗時資料寫入 logs/timing_log_pipeline_direct.csv，執行前會先清空。

使用方式：
    python3 test_api_pipeline_direct.py [圖片路徑或資料夾路徑]

範例：
    python3 test_api_pipeline_direct.py
    python3 test_api_pipeline_direct.py /home/ubuntu/Documents/API_Test/test_images/v1/sample.jpg
    python3 test_api_pipeline_direct.py /home/ubuntu/Documents/API_Test/test_images/v1_release
"""

import csv
import sys
import time
from pathlib import Path

from service_api.pipeline import BoxDetectionPipeline

DEFAULT_IMAGE_PATH = "/home/ubuntu/Documents/API_Test/test_images/博亞藥局/"

REPO_ROOT       = Path(__file__).resolve().parent
TIMING_LOG_PATH = REPO_ROOT / "logs" / "timing_log_pipeline_direct.csv"
IMAGE_SUFFIXES  = (".jpg", ".jpeg", ".png")


def process_image(img_path: Path, pipeline: BoxDetectionPipeline) -> dict:
    """對單張圖片直接呼叫 pipeline.run()，回傳耗時統計。"""
    image_bytes = img_path.read_bytes()

    t0     = time.time()
    result = pipeline.run(image_bytes, img_path.name)
    total_time = time.time() - t0

    return {
        "filename":     img_path.name,
        "num_boxes":    result.total_boxes,
        "total_time_s": total_time,
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

    pipeline = BoxDetectionPipeline()

    TIMING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TIMING_LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "num_boxes", "total_time_s"])

        for img_path in images:
            print(f"Processing {img_path.name} ...")
            row = process_image(img_path, pipeline)
            print(f"  boxes={row['num_boxes']} total={row['total_time_s']:.3f}s")
            writer.writerow([
                row["filename"],
                row["num_boxes"],
                f"{row['total_time_s']:.3f}",
            ])


def print_timing_table() -> None:
    """讀取 logs/timing_log_pipeline_direct.csv，印出整理好的表格。"""
    if not TIMING_LOG_PATH.exists():
        print(f"找不到 timing log：{TIMING_LOG_PATH}")
        return

    with open(TIMING_LOG_PATH, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print("timing log 是空的。")
        return

    box_sum   = sum(int(r["num_boxes"]) for r in rows)
    total_sum = sum(float(r["total_time_s"]) for r in rows)
    n = len(rows)

    header = f"{'圖片':<16}{'箱數':<6}{'總花費時間 (s)':<16}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['filename']:<16}{r['num_boxes']:<6}{r['total_time_s']:<16}")
    print("-" * len(header))
    print(f"{'平均':<16}{box_sum/n:<6.1f}{total_sum/n:<16.3f}")


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_IMAGE_PATH)
    run(image_path)
    print_timing_table()


if __name__ == "__main__":
    main()
