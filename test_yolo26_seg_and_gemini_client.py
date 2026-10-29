"""
test_yolo26_seg_and_gemini_client.py — YOLO26 segmentation + 服務 GeminiBoxClient 整合測試腳本
==============================================================================================
整合 test_yolo26_seg.py 與 test_gemini_client.py，並照服務 pipeline.py 的規則合併兩者結果：
  1. YOLO segmentation（服務的模型與參數）取得每個紙箱的 mask polygon
  2. 原圖 bytes 直接傳給 GeminiBoxClient.detect，取得每個紙箱的品名/日期/box_2d
  3. 對每筆 Gemini 結果，計算 box_2d 中心點：
       若中心點落在某個 YOLO mask 內 → 保留該 mask + Gemini 資訊
       若中心點沒有落在任何 mask 內 → 捨棄
     （比對直接呼叫 pipeline._find_containing_polygon_index，與服務完全一致）

使用方式：
    python3 test_yolo26_seg_and_gemini_client.py --source <圖片或資料夾> [--out 輸出資料夾]

範例：
    python3 test_yolo26_seg_and_gemini_client.py --source sample_images
    python3 test_yolo26_seg_and_gemini_client.py --source "sample_images/全家/正面-2.jpg"

資料夾會遞迴搜尋子資料夾，輸出保留相對路徑（預設 results/yolo26_seg_gemini/）：
    yolo26_seg/<相對路徑>/<圖片檔名>_seg.jpg     YOLO mask + #編號 信心值
    gemini/<相對路徑>/<圖片檔名>_gemini.jpg      Gemini box_2d 矩形 + 品名/日期 + 紅色中心點
    final/<相對路徑>/<圖片檔名>_final.jpg        保留的 YOLO mask + Gemini 品名/日期 + 紅色中心點
    summary.json                                 每張圖的 YOLO / Gemini / 合併結果與統計
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from service_api import config
from service_api.pipeline import _find_containing_polygon_index
from service_api.services.gemini_client import GeminiBoxClient
from test_gemini_client import box_2d_to_result, draw_centers
from test_gemini_contour import PALETTE, draw_result
from test_yolo26_seg import IMAGE_SUFFIXES, draw as draw_yolo


def list_images(source: Path) -> list[Path]:
    """單張圖片或資料夾（含子資料夾）內的所有圖片。"""
    if source.is_file():
        return [source]
    return sorted(p for p in source.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)


def merge(polygons: list[np.ndarray], gemini_boxes: list[dict], w: int, h: int) -> tuple[list[dict], list[dict]]:
    """
    依服務 pipeline 規則合併：每筆 Gemini 的 box_2d 中心點落在哪個 YOLO mask 內就採用該 mask。

    Returns:
        (kept, dropped)
        kept:    draw_result 格式 + yolo_index / gemini_id
        dropped: 被捨棄的 Gemini 結果（中心點不在任何 mask 內）
    """
    kept, dropped = [], []
    for gid, gbox in enumerate(gemini_boxes):
        g = box_2d_to_result(gid, gbox, w, h)
        cx, cy = g["center"]
        index = _find_containing_polygon_index(polygons, cx, cy)
        if index is None:
            dropped.append({"gemini_id": gid, "center": [round(cx, 1), round(cy, 1)], **gbox})
            continue
        i = len(kept)
        kept.append({
            **g,
            "id": i,
            "color": PALETTE[i % len(PALETTE)][0],
            "polygon": polygons[index].tolist(),
            "yolo_index": index,
            "gemini_id": gid,
        })
    return kept, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO26 segmentation + GeminiBoxClient 整合測試")
    parser.add_argument("--model",  default=config.YOLO_MODEL_PATH, help="模型權重 .pt")
    parser.add_argument("--source", required=True, help="圖片檔或資料夾（遞迴搜尋子資料夾）")
    parser.add_argument("--out",    help="輸出資料夾（預設 results/yolo26_seg_gemini）")
    parser.add_argument("--conf",   type=float, default=config.YOLO_CONF_THRESHOLD, help="YOLO 信心值門檻")
    parser.add_argument("--imgsz",  type=int,   default=config.YOLO_IMG_SIZE, help="YOLO 推論縮放邊長")
    parser.add_argument("--device", default=None, help="例如 0、cpu；不給由 ultralytics 自動選")
    args = parser.parse_args()

    source = Path(args.source)
    images = list_images(source)
    if not images:
        raise SystemExit(f"找不到圖片：{source}")
    base = source if source.is_dir() else source.parent

    out_dir = Path(args.out) if args.out else config.ROOT_DIR / "results" / "yolo26_seg_gemini"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    client = GeminiBoxClient()
    print(f"YOLO：{args.model}  conf={args.conf} imgsz={args.imgsz} device={args.device or 'auto'}")
    print(f"Gemini：{config.GEMINI_MODEL}")
    print(f"圖片：{len(images)} 張 → 輸出到 {out_dir}\n")

    # 暖機一次，避免第一張圖的推論時間包含 CUDA 初始化
    model.predict(np.zeros((args.imgsz, args.imgsz, 3), np.uint8), imgsz=args.imgsz, device=args.device, verbose=False)

    summary = []
    for path in images:
        rel_dir = path.parent.relative_to(base)
        image_bytes = path.read_bytes()
        img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)  # 同服務 pipeline
        if img is None:
            print(f"⚠️ 無法讀取 {path}，略過")
            continue
        h, w = img.shape[:2]
        print(f"── {path.relative_to(base)}  {w}x{h}")

        # ── YOLO ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        yolo = model.predict(img, task="seg", imgsz=args.imgsz, conf=args.conf, device=args.device, verbose=False)[0]
        yolo_ms = (time.perf_counter() - t0) * 1000
        if yolo.masks is None:
            polygons, confs, yolo_boxes = [], [], np.zeros((0, 4))
        else:
            polygons = yolo.masks.xy
            confs = yolo.boxes.conf.cpu().numpy().tolist()
            yolo_boxes = yolo.boxes.xyxy.cpu().numpy()
        seg_path = out_dir / "yolo26_seg" / rel_dir / f"{path.stem}_seg.jpg"
        seg_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(seg_path), draw_yolo(img, polygons, confs, yolo_boxes))

        # ── Gemini ────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            gemini_boxes = client.detect(image_bytes, filename=path.name)
        except Exception as e:
            print(f"⚠️ Gemini 呼叫失敗：{e}\n")
            summary.append({"image": str(path.relative_to(base)), "width": w, "height": h, "error": str(e)})
            continue
        gemini_sec = time.perf_counter() - t0
        gemini_result = [box_2d_to_result(i, b, w, h) for i, b in enumerate(gemini_boxes)]
        gemini_path = out_dir / "gemini" / rel_dir / f"{path.stem}_gemini.jpg"
        gemini_path.parent.mkdir(parents=True, exist_ok=True)
        draw_centers(draw_result(img, gemini_result), gemini_result).save(gemini_path, quality=95)

        # ── 合併（服務 pipeline 規則）───────────────────────────────────────────
        kept, dropped = merge(polygons, gemini_boxes, w, h)
        final_path = out_dir / "final" / rel_dir / f"{path.stem}_final.jpg"
        final_path.parent.mkdir(parents=True, exist_ok=True)
        draw_centers(draw_result(img, kept), kept).save(final_path, quality=95)

        used = [k["yolo_index"] for k in kept]
        unused_yolo = [i for i in range(len(polygons)) if i not in used]
        shared_yolo = sorted({i for i in used if used.count(i) > 1})
        print(f"   YOLO {len(polygons):>2} ({yolo_ms:.0f} ms)   Gemini {len(gemini_boxes):>2} ({gemini_sec:.1f} s)   "
              f"保留 {len(kept):>2}   捨棄 {len(dropped):>2}   YOLO 未配對 {len(unused_yolo):>2}   "
              f"mask 被重複使用 {len(shared_yolo)}\n")

        summary.append({
            "image": str(path.relative_to(base)),
            "width": w,
            "height": h,
            "yolo_ms": round(yolo_ms, 1),
            "gemini_sec": round(gemini_sec, 2),
            "num_yolo": len(polygons),
            "num_gemini": len(gemini_boxes),
            "num_kept": len(kept),
            "num_dropped": len(dropped),
            "unused_yolo_indices": unused_yolo,    # YOLO 有框到、但沒有任何 Gemini 中心點落入
            "shared_yolo_indices": shared_yolo,    # 同一個 mask 被多筆 Gemini 結果使用（會重複計數）
            "kept": [
                {
                    "id": k["id"],
                    "gemini_id": k["gemini_id"],
                    "yolo_index": k["yolo_index"],
                    "yolo_conf": round(confs[k["yolo_index"]], 4),
                    "product_name": k["name"],
                    "manufacturer_date": k["manufacturer_date"],
                    "expiration_date": k["expiration_date"],
                    "center": [round(v, 1) for v in k["center"]],
                    "box_2d": gemini_boxes[k["gemini_id"]]["box_2d"],
                }
                for k in kept
            ],
            "dropped": dropped,
        })

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "yolo_model": args.model, "conf": args.conf, "imgsz": args.imgsz,
        "gemini_model": config.GEMINI_MODEL, "images": summary,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [s for s in summary if "error" not in s]
    totals = {k: sum(s[k] for s in ok) for k in ("num_yolo", "num_gemini", "num_kept", "num_dropped")}
    unused = sum(len(s["unused_yolo_indices"]) for s in ok)
    print(f"共 {len(ok)}/{len(summary)} 張成功：YOLO {totals['num_yolo']}、Gemini {totals['num_gemini']}、"
          f"保留 {totals['num_kept']}、捨棄 {totals['num_dropped']}、YOLO 未配對 {unused}")
    print(f"結果已存到 {out_dir}")


if __name__ == "__main__":
    main()
