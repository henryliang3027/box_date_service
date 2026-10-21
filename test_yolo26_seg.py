"""
test_yolo26_seg.py — YOLO26 紙箱 segmentation 模型測試腳本
==========================================================
只跑 YOLO segmentation（不呼叫 Gemini），把每張圖的 mask、信心值畫在圖上存檔，
並輸出 summary.json（每張圖的紙箱數、信心值、bbox、mask 輪廓），方便檢查模型本身的表現。

推論參數預設與服務一致（service_api/config.py 的 YOLO_IMG_SIZE、YOLO_CONF_THRESHOLD）。

使用方式：
    python3 test_yolo26_seg.py --source <圖片或資料夾> [--out 輸出資料夾] [--conf 0.7] [--imgsz 640] [--device 0]

範例：
    python3 test_yolo26_seg.py --source /home/ubuntu/Documents/API_Test/test_images/全家
    python3 test_yolo26_seg.py --source sample_images/resized.jpg --conf 0.5
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from service_api import config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 每個 instance 的顏色（BGR），依序循環
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207),
    (10, 249, 72), (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0),
    (168, 153, 44), (255, 194, 0), (147, 69, 52), (255, 115, 100), (236, 24, 0),
    (255, 56, 132), (133, 0, 82), (255, 56, 203), (200, 149, 255), (199, 55, 255),
]


def list_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    return sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def _mask_center(poly: np.ndarray, box: np.ndarray) -> tuple[float, float]:
    """mask 的重心（面積中心）；輪廓退化（面積為 0）時改用 bbox 中心。"""
    if len(poly) >= 3:
        m = cv2.moments(poly.astype(np.float32))
        if m["m00"] > 0:
            return m["m10"] / m["m00"], m["m01"] / m["m00"]
    return (box[0] + box[2]) / 2, (box[1] + box[3]) / 2


def draw(img: np.ndarray, polygons: list[np.ndarray], confs: list[float], boxes: np.ndarray) -> np.ndarray:
    """mask 半透明填色 + 輪廓線 + 「#編號 信心值」標籤，左上角寫總數。"""
    h, w = img.shape[:2]
    scale = max(w, h) / 1000                       # 線寬 / 字級依圖片大小縮放
    thick = max(2, round(2 * scale))
    font_scale = max(0.6, 0.7 * scale)
    font_thick = max(1, round(1.5 * scale))

    overlay = img.copy()
    for i, poly in enumerate(polygons):
        if len(poly) >= 3:
            cv2.fillPoly(overlay, [poly.astype(np.int32)], PALETTE[i % len(PALETTE)])
    out = cv2.addWeighted(overlay, config.MASK_ALPHA, img, 1 - config.MASK_ALPHA, 0)

    for i, (poly, conf, box) in enumerate(zip(polygons, confs, boxes)):
        color = PALETTE[i % len(PALETTE)]
        if len(poly) >= 3:
            cv2.polylines(out, [poly.astype(np.int32)], True, color, thick)
        label = f"#{i + 1} {conf:.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        cx, cy = _mask_center(poly, box)
        # 標籤框以 mask 中心為中心，超出圖片時往內貼齊
        bw, bh = tw + 8, th + base + 6
        x1 = int(min(max(0, cx - bw / 2), w - bw))
        y1 = int(min(max(0, cy - bh / 2), h - bh))
        cv2.rectangle(out, (x1, y1), (x1 + bw, y1 + bh), color, -1)
        cv2.putText(out, label, (x1 + 4, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)

    total = f"Total: {len(polygons)}"
    (tw, th), base = cv2.getTextSize(total, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.4, font_thick + 1)
    cv2.rectangle(out, (0, 0), (tw + 16, th + base + 16), (0, 0, 0), -1)
    cv2.putText(out, total, (8, th + 8), cv2.FONT_HERSHEY_SIMPLEX, font_scale * 1.4,
                (255, 255, 255), font_thick + 1, cv2.LINE_AA)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="測試 YOLO26 紙箱 segmentation 模型")
    parser.add_argument("--model",  default=config.YOLO_MODEL_PATH, help="模型權重 .pt")
    parser.add_argument("--source", required=True, help="圖片檔或資料夾")
    parser.add_argument("--out",    help="輸出資料夾（預設 results/yolo26_seg/<來源資料夾名稱>）")
    parser.add_argument("--conf",   type=float, default=config.YOLO_CONF_THRESHOLD, help="信心值門檻")
    parser.add_argument("--imgsz",  type=int,   default=config.YOLO_IMG_SIZE, help="推論縮放邊長")
    parser.add_argument("--device", default=None, help="例如 0、cpu；不給由 ultralytics 自動選")
    args = parser.parse_args()

    source = Path(args.source)
    images = list_images(source)
    if not images:
        raise SystemExit(f"找不到圖片：{source}")

    folder_name = source.name if source.is_dir() else source.parent.name
    out_dir = Path(args.out) if args.out else config.ROOT_DIR / "results" / "yolo26_seg" / folder_name
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    print(f"模型：{args.model}")
    print(f"參數：conf={args.conf} imgsz={args.imgsz} device={args.device or 'auto'}")
    print(f"圖片：{len(images)} 張 → 輸出到 {out_dir}\n")

    # 暖機一次，避免第一張圖的推論時間包含 CUDA 初始化
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), imgsz=args.imgsz, device=args.device, verbose=False)

    summary = []
    for path in images:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)   # 與服務的 cv2.imdecode 相同，會套用 EXIF 方向
        if img is None:
            print(f"⚠️ 無法讀取 {path.name}，略過")
            continue
        h, w = img.shape[:2]

        t0 = time.perf_counter()
        result = model.predict(img, task="seg", imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000

        if result.masks is None:
            polygons, confs, boxes = [], [], np.zeros((0, 4))
        else:
            polygons = result.masks.xy
            confs = result.boxes.conf.cpu().numpy().tolist()
            boxes = result.boxes.xyxy.cpu().numpy()

        out_path = out_dir / f"{path.stem}_seg.jpg"
        cv2.imwrite(str(out_path), draw(img, polygons, confs, boxes))

        conf_str = ", ".join(f"{c:.2f}" for c in confs) or "-"
        print(f"{path.name:<20} {w}x{h:<6} 紙箱 {len(polygons):>2}   {ms:6.1f} ms   conf: {conf_str}")

        summary.append({
            "image": path.name,
            "width": w,
            "height": h,
            "num_boxes": len(polygons),
            "inference_ms": round(ms, 1),
            "boxes": [
                {
                    "id": i + 1,
                    "conf": round(c, 4),
                    "bbox_xyxy": [round(float(v), 1) for v in b],
                    "mask": [[round(float(x) / w, 4), round(float(y) / h, 4)] for x, y in p],  # 正規化 0~1，同 API
                }
                for i, (p, c, b) in enumerate(zip(polygons, confs, boxes))
            ],
        })

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "model": args.model, "conf": args.conf, "imgsz": args.imgsz, "images": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(s["num_boxes"] for s in summary)
    avg_ms = sum(s["inference_ms"] for s in summary) / max(1, len(summary))
    print(f"\n共 {len(summary)} 張、{total} 個紙箱，平均推論 {avg_ms:.1f} ms")
    print(f"標注圖與 summary.json 已存到 {out_dir}")


if __name__ == "__main__":
    main()
