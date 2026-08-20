"""
services/detector.py — YOLO 紙箱偵測封裝
==========================================
將 ultralytics YOLO 的載入與推論包成 BoxDetector 類別。
呼叫端只需傳入圖片（numpy array），不需知道 YOLO API 細節。

設計原則：
  - 模型只在 __init__ 時載入一次（避免每次請求重新載入）
  - 回傳 DetectionResult 物件（而非 ultralytics 內部物件），
    讓 pipeline 與 ultralytics 解耦，未來換模型只需改這一個檔案
"""

import numpy as np                  # numpy array 是 YOLO 原生支援的輸入格式
from ultralytics import YOLO        # YOLO 推論框架

from service_api import config      # 統一設定（模型路徑、信心值閾值等）


class BoxDetector:
    """
    封裝 YOLO segmentation 推論。

    使用方式：
        detector = BoxDetector()
        result = detector.detect(img_bgr)  # img_bgr 是 OpenCV BGR numpy array
    """

    def __init__(self) -> None:
        # 載入 YOLO 模型（首次載入需幾秒，之後常駐記憶體供後續請求複用）
        self._model = YOLO(config.YOLO_MODEL_PATH)

    def detect(self, img_bgr: np.ndarray) -> "DetectionResult":
        """
        對一張圖片執行 YOLO segmentation 推論。

        Args:
            img_bgr: OpenCV 格式的圖片（BGR，shape=(H, W, 3)）

        Returns:
            DetectionResult（包含每個 box 的 polygon 輪廓點與信心值）
        """
        # 執行推論，verbose=False 抑制 YOLO 的 console 進度輸出
        results = self._model.predict(
            img_bgr,                           # 直接傳 numpy array（不需存成暫存檔）
            task="seg",                        # segmentation 任務（取得 mask polygon）
            imgsz=config.YOLO_IMG_SIZE,        # 推論時的縮放邊長
            conf=config.YOLO_CONF_THRESHOLD,   # 低於此信心值的結果自動過濾
            verbose=False,                     # 不印 YOLO 的 console log
        )

        result = results[0]  # 單張圖片只會有一個結果物件

        # 沒有偵測到任何 mask 時，回傳空結果（不 crash）
        if result.masks is None:
            return DetectionResult(polygons=[], confs=[])

        # result.masks.xy：每個 mask 的輪廓點，list of float (N,2) ndarray
        polygons: list[np.ndarray] = result.masks.xy
        # result.boxes.conf：信心值 tensor，.cpu().numpy() 轉為 Python list
        confs: list[float] = result.boxes.conf.cpu().numpy().tolist()

        return DetectionResult(polygons=polygons, confs=confs)


class DetectionResult:
    """
    YOLO 推論結果的輕量容器。

    以自訂類別而非 ultralytics 內部物件傳遞結果，
    讓呼叫端不需 import ultralytics，降低模組間的耦合。
    """

    def __init__(
        self,
        polygons: list[np.ndarray],  # 每個 box 的 mask 輪廓點
        confs: list[float],          # 對應的信心值（與 polygons 等長）
    ) -> None:
        self.polygons = polygons          # mask 輪廓點列表
        self.confs    = confs             # 信心值列表
        self.total    = len(polygons)     # 偵測到的 box 總數（方便直接存取）
