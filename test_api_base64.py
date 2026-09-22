"""
test_api_base64.py — /api/v1/detect/image_base64 功能測試腳本
==================================================================
把圖片編碼成 Base64，呼叫 /api/v1/detect/image_base64（include_image=true），
把回應內的標注圖（annotated_image_base64）解碼存到 results/ 資料夾，
並印出偵測到的紙箱數量、品名、日期等 JSON 結果。

使用方式：
    python3 test_api_base64.py [圖片路徑] [API base URL]

範例：
    python3 test_api_base64.py
    python3 test_api_base64.py sample_images/resized.jpg
    python3 test_api_base64.py sample_images/resized.jpg http://localhost:8080
"""

import base64
import json
import sys
from pathlib import Path

import requests

DEFAULT_IMAGE_PATH = "sample_images/resized.jpg"
DEFAULT_API_BASE   = "http://127.0.0.1:8081"

REPO_ROOT   = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / "results"


def call_detect_image_base64(image_path: Path, api_base: str) -> dict:
    """把圖片編碼成 base64，呼叫 /api/v1/detect/image_base64，回傳解析後的 JSON。"""
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    resp = requests.post(
        f"{api_base}/api/v1/detect/image_base64",
        json={"image_base64": image_base64, "include_image": True},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def save_annotated_image(data: dict, image_path: Path) -> Path | None:
    """把回應內的 annotated_image_base64 解碼存到 results/。"""
    annotated_b64 = data.get("annotated_image_base64")
    if not annotated_b64:
        print("⚠️  回應中沒有 annotated_image_base64（可能沒偵測到任何紙箱）。")
        return None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{image_path.stem}_annotated.jpg"
    out_path.write_bytes(base64.b64decode(annotated_b64))
    return out_path


def print_summary(data: dict) -> None:
    print(f"\nfilename    : {data.get('filename')}")
    print(f"total_boxes : {data.get('total_boxes')}")
    for box in data.get("boxes", []):
        product = box.get("product") or {}
        brand   = product.get("brand") or ""
        name    = product.get("name") or "未知品項"
        expiry  = box.get("expiry_date")
        expiry_str = f"{expiry['year']}-{expiry['month']}-{expiry['day']}" if expiry else "無法解析"
        print(f"  box_id={box['box_id']:<3} product={brand}{name}  expiry_date={expiry_str}")


def main() -> None:
    image_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_IMAGE_PATH)
    api_base   = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_API_BASE

    if not image_path.exists():
        print(f"❌ 找不到圖片：{image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🚀 呼叫 {api_base}/api/v1/detect/image_base64（圖片：{image_path.name}）")
    data = call_detect_image_base64(image_path, api_base)

    print_summary(data)

    out_path = save_annotated_image(data, image_path)
    if out_path:
        print(f"\n💾 標注圖已存到：{out_path}")


if __name__ == "__main__":
    main()
