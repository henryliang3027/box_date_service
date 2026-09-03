"""
pipeline.py — 核心業務流程協調者
==================================
將 YOLO segmentation 與 Gemini 整圖辨識結果整合成一個完整流程。

API route 只需呼叫 pipeline.run()，不需知道各服務的細節。
每新增一個處理步驟，只需在這裡修改，不影響 main.py（路由層）。

流程圖：
  上傳圖片
    → (1) YOLO 偵測 → 取得所有紙箱的 mask polygon
    → (2) Gemini 對整張圖推論 → 取得所有紙箱的
           product_name / manufacturer_date / expiration_date / box_2d
    → (3) 對每筆 Gemini 結果，計算 box_2d 中心點；
           若中心點落在某個 YOLO mask 內 → 保留該 mask + Gemini 資訊
           若中心點沒有落在任何 mask 內 → 捨棄
    → (4) 將 Gemini 的 product_name 與 box_name.json 資料庫做模糊比對，
           取得正確的 brand / name（比對不到則 brand=None，name 沿用原始文字）
    → (5) 組合回應（含選用的標注圖片）
"""

import base64                   # 將標注圖片編碼為 base64 字串
import time                     # 計算 YOLO / Gemini 耗時（DEBUG 用）
from pathlib import Path        # 計算 timing log 檔案路徑

import cv2                      # 解碼上傳的圖片 bytes、point-in-polygon 判斷
import numpy as np              # numpy array 操作

from service_api.schemas import (   # Pydantic 回應格式
    BoxResult,
    DateInfo,
    DetectResponse,
    ProductInfo,
)
from service_api import config
from service_api.services.box_matcher    import match_product    # 品名資料庫模糊比對
from service_api.services.detector       import BoxDetector      # YOLO 偵測器
from service_api.services.gemini_client  import GeminiBoxClient  # Gemini 整圖辨識
from service_api.utils.image_utils       import (
    draw_results,             # 在原圖上繪製偵測結果
    pil_to_bytes,             # 將 PIL Image 轉為 bytes
)


class BoxDetectionPipeline:
    """
    紙箱偵測完整流程的協調者（Orchestrator Pattern）。

    此類別持有所有服務的引用，由 FastAPI lifespan 在啟動時建立，
    整個應用程式生命週期只建立一次（避免重複載入 YOLO 模型）。

    使用方式：
        pipeline = BoxDetectionPipeline()          # startup 時呼叫一次
        result   = pipeline.run(img_bytes, fname)  # 每次請求呼叫
    """

    def __init__(self) -> None:
        # 初始化 YOLO 偵測器（會在此時載入模型，需幾秒）
        self.detector = BoxDetector()
        # 初始化 Gemini 客戶端（不載入模型，呼叫外部 API）
        self.gemini   = GeminiBoxClient()

    def run(
        self,
        image_bytes:      bytes,        # 上傳的圖片原始 bytes
        filename:         str,          # 原始檔名（放進回應供識別）
        include_image:    bool = False, # 是否在回應中附上標注圖片
        print_confidence: bool = False, # True 時，逐一印出命中 box 的 YOLO confidence
    ) -> DetectResponse:
        """
        對上傳圖片執行完整偵測流程，回傳結構化結果。

        Args:
            image_bytes:   上傳圖片的原始 bytes（JPG/PNG）
            filename:      原始檔名
            include_image: True 時回應會附上 base64 編碼的標注圖片

        Returns:
            DetectResponse（所有 box 的結果 + 可選的標注圖片）
        """
        print(f"filename={filename}")
        # ── 步驟 1：將 bytes 轉為 OpenCV BGR numpy array ────────────────────
        nparr   = np.frombuffer(image_bytes, np.uint8)    # bytes → uint8 array
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # 解碼為 BGR 三通道圖
        img_h, img_w = img_bgr.shape[:2]                  # 圖片高寬（mask 正規化 / box_2d 換算用）

        # ── 步驟 2：YOLO 推論，取得所有紙箱的 mask polygon ──────────────────
        yolo_t0   = time.time()
        detection = self.detector.detect(img_bgr)          # 傳入 BGR numpy array
        yolo_time = time.time() - yolo_t0
        if config.DEBUG:
            print(f"[DEBUG] YOLO segmentation time: {yolo_time:.3f}s  boxes={detection.total}")

        # ── 步驟 3：Gemini 對整張圖推論，取得所有紙箱的品名/日期/box_2d ──────
        gemini_t0    = time.time()
        gemini_boxes = self.gemini.detect(image_bytes, filename=filename)
        gemini_time  = time.time() - gemini_t0
        if config.DEBUG:
            print(f"[DEBUG] Gemini inference time: {gemini_time:.3f}s  boxes={len(gemini_boxes)}")

        _log_timing(filename, yolo_time, gemini_time)

        # ── 步驟 4：逐一比對每筆 Gemini 結果與 YOLO mask ────────────────────
        box_results:       list[BoxResult]   = []  # 收集每個 box 的結構化結果
        matched_polygons:  list[np.ndarray]  = []  # 有命中的 YOLO mask（繪圖用）
        labels:            list[str]         = []  # 同步收集文字標籤（繪圖用）

        box_id = 0
        for gbox in gemini_boxes:
            # 4a. 將 box_2d（0-1000 正規化，[ymin, xmin, ymax, xmax]）換算成原圖像素座標的中心點
            ymin, xmin, ymax, xmax = gbox["box_2d"]
            cx = (xmin + xmax) / 2 / 1000 * img_w
            cy = (ymin + ymax) / 2 / 1000 * img_h

            # 4b. 找出中心點落在哪個 YOLO mask 內；沒有命中就捨棄這筆 Gemini 結果
            matched_index = _find_containing_polygon_index(detection.polygons, cx, cy)
            if matched_index is None:
                continue
            matched_polygon = detection.polygons[matched_index]
            matched_conf    = detection.confs[matched_index]

            box_id += 1
            if print_confidence:
                print(f"[CONFIDENCE] box_id={box_id} yolo_confidence={matched_conf:.4f}")

            # 4c. mask 輪廓點正規化（除以圖片寬高，值域 0.0~1.0，與圖片尺寸無關）
            normalized_mask = [
                [round(float(x) / img_w, 4), round(float(y) / img_h, 4)]
                for x, y in matched_polygon
            ]

            matched      = match_product(gbox.get("product_name"))
            brand_name   = matched["brand"]
            product_name = matched["name"] or "未知品項"
            expiry       = _gemini_date_to_info(gbox.get("expiration_date"))
            mfg          = _gemini_date_to_info(gbox.get("manufacturer_date"))

            box_results.append(BoxResult(
                box_id           = box_id,
                mask             = normalized_mask,
                product          = ProductInfo(brand_name=brand_name, product_name=product_name),
                expiry_date      = expiry,
                manufacture_date = mfg,
            ))
            matched_polygons.append(matched_polygon)

            # 4d. 建立文字標籤（box_id + 品名 + 日期，多行顯示）
            expiry_str = f"{expiry.year}/{expiry.month}/{expiry.day}" if expiry else "無"
            mfg_str    = f"{mfg.year}/{mfg.month}/{mfg.day}" if mfg else "無"
            date_lines = [f"有效日期：{expiry_str}", f"製造日期：{mfg_str}"]
            name_line = f"#{box_id} [{brand_name}] {product_name}" if brand_name else f"#{box_id} {product_name}"
            labels.append("\n".join([name_line] + date_lines))

        # ── 步驟 5：視覺化（選用）─────────────────────────────────────────────
        annotated_b64: str | None = None
        if include_image and matched_polygons:  # 有命中的 box 才繪製
            result_img = draw_results(img_bgr, matched_polygons, labels)
            annotated_b64 = base64.b64encode(pil_to_bytes(result_img)).decode("utf-8")

        # ── 步驟 6：組合並回傳完整回應 ───────────────────────────────────────
        return DetectResponse(
            filename                = filename,
            total_boxes             = len(box_results),
            boxes                   = box_results,
            annotated_image_base64  = annotated_b64,
        )


# ── 內部輔助函式 ──────────────────────────────────────────────────────────────

_TIMING_LOG_PATH = str(Path(__file__).resolve().parent.parent / "logs" / "timing_log.csv")


def _log_timing(filename: str, yolo_time: float, gemini_time: float) -> None:
    """把每次請求的 YOLO / Gemini 耗時附加寫入 CSV，供效能分析用。"""
    import csv
    import os

    os.makedirs(os.path.dirname(_TIMING_LOG_PATH), exist_ok=True)
    file_exists = os.path.exists(_TIMING_LOG_PATH)
    with open(_TIMING_LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["filename", "yolo_time_s", "gemini_time_s", "total_time_s"])
        writer.writerow([filename, f"{yolo_time:.3f}", f"{gemini_time:.3f}", f"{yolo_time + gemini_time:.3f}"])


def _find_containing_polygon_index(
    polygons: list[np.ndarray],  # 所有 YOLO mask polygon
    cx: float,                   # Gemini box_2d 中心點 x（原圖像素座標）
    cy: float,                   # Gemini box_2d 中心點 y（原圖像素座標）
) -> int | None:
    """
    找出「包含中心點 (cx, cy)」的第一個 mask polygon 的索引。

    Returns:
        命中的 polygon 在 polygons 中的索引，沒有任何 mask 包含該點則回傳 None。
    """
    for index, polygon in enumerate(polygons):
        pts = polygon.astype(np.float32)
        # cv2.pointPolygonTest：>=0 代表點在多邊形內部或邊界上
        if cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0:
            return index
    return None


def _gemini_date_to_info(d: dict | None) -> DateInfo | None:
    """
    將 Gemini 回傳的日期 dict（{"year","month","day"} 或 None）轉為 Pydantic DateInfo。
    任一欄位缺漏、或值為 "未知" 時視為無法解析，回傳 None。
    """
    if not d or not isinstance(d, dict):
        return None
    year, month, day = d.get("year"), d.get("month"), d.get("day")
    if not year or not month or not day:
        return None
    if "未知" in (year, month, day):
        return None
    return DateInfo(year=year, month=month, day=day)
