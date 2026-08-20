"""
utils/image_utils.py — 影像處理工具函式
========================================
提供兩個核心功能，供 pipeline.py 使用：

  crop_with_mask(img_bgr, polygon)
    給定原圖與 mask polygon，裁切出去背後的 box 區域。

  draw_results(img_bgr, polygons, labels)
    在原圖上疊加所有 box 的半透明 mask、輪廓線、文字標籤。

  pil_to_bytes(image, fmt)
    將 PIL Image 轉為 bytes（用於 API 回傳圖片串流）。
"""

import io        # 記憶體緩衝區（pil_to_bytes 用）
import random    # 隨機顏色生成

import cv2                                     # OpenCV：mask 運算、影像解碼
import numpy as np                             # numpy array 操作
from PIL import Image, ImageDraw, ImageEnhance, ImageFont    # PIL：中文文字繪製

from service_api import config                 # 字型路徑、透明度等視覺化設定


def crop_with_mask(img_bgr: np.ndarray, polygon: np.ndarray) -> Image.Image:
    """
    裁切指定 mask polygon 的區域，並將 polygon 外的背景設為黑色。

    流程：
      1. 以 polygon 建立 binary mask（polygon 內=255，外=0）
      2. bitwise_and 將背景歸零（去背）
      3. 取 polygon 的最小外接矩形作為裁切範圍

    Args:
        img_bgr: OpenCV 格式原圖（BGR，shape=(H, W, 3)）
        polygon: mask 輪廓點，shape=(N, 2)，float32

    Returns:
        裁切後去背的 PIL RGB Image，尺寸為 mask 的最小外接矩形
    """
    h, w = img_bgr.shape[:2]              # 取原圖高寬（用來建立同尺寸 mask）
    pts  = polygon.astype(np.int32)        # polygon 座標轉整數（fillPoly 需要）

    # 建立與原圖同尺寸的單通道遮罩（全黑=0）
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)         # 將 polygon 內部填白（255=前景）

    # 套用遮罩：mask=0 的像素在 bitwise_and 後變成 0（黑色），達到去背效果
    masked = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)

    # boundingRect 回傳 polygon 的最小外接矩形 (x, y, width, height)
    x, y, bw, bh = cv2.boundingRect(pts)
    cropped = masked[y : y + bh, x : x + bw]  # numpy slicing 裁切矩形區域

    # OpenCV 是 BGR，PIL 是 RGB，需要色彩空間轉換
    return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))


def draw_results(
    img_bgr:  np.ndarray,
    polygons: list[np.ndarray],
    labels:   list[str],
) -> Image.Image:
    """
    在原圖上繪製所有 box 的視覺化結果（三層疊加）：
      第一層：半透明色塊（讓紙箱區域更顯眼）
      第二層：mask 輪廓線（明確標示邊界）
      第三層：文字標籤（品名 + 日期，可換行）

    Args:
        img_bgr:  OpenCV 格式原圖（BGR，shape=(H, W, 3)）
        polygons: 所有 box 的 mask 輪廓點列表（與 labels 等長）
        labels:   每個 box 的顯示文字（可含 \\n 換行）

    Returns:
        繪製完成的 PIL RGB Image
    """
    # 為每個 box 隨機產生互不相同的顏色（BGR 格式）
    colors = _random_colors(len(polygons))

    # ── 第一層：半透明 mask 色塊 ─────────────────────────────────────────────
    overlay = img_bgr.copy()          # 複製一份原圖，用來畫填色（不改動原圖）
    for pts, color in zip(polygons, colors):
        cv2.fillPoly(overlay, [pts.astype(np.int32)], color)  # 填滿 polygon 區域

    # addWeighted：overlay * alpha + img_bgr * (1-alpha) → 半透明效果
    blended = cv2.addWeighted(
        overlay,   config.MASK_ALPHA,          # 填色圖佔 40%
        img_bgr,   1 - config.MASK_ALPHA,      # 原圖佔 60%
        0,                                     # gamma 補正值（0=不補正）
    )

    # ── 第二層：mask 輪廓線 ───────────────────────────────────────────────────
    for pts, color in zip(polygons, colors):
        cv2.polylines(
            blended,
            [pts.astype(np.int32)],   # 輪廓點（需為 list of arrays）
            isClosed=True,            # 首尾相連，形成封閉輪廓
            color=color,              # 與填色同色，視覺上一致
            thickness=2,              # 線寬 2px
        )

    # ── 第三層：文字標籤（PIL 才能顯示中文，OpenCV putText 不支援）────────────
    # 先將 OpenCV BGR 圖轉為 PIL RGB，才能用 PIL 繪製中文
    pil_img   = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    draw      = ImageDraw.Draw(pil_img)                  # 建立 PIL 繪圖物件
    font_size = _dynamic_font_size(img_bgr.shape)     # 依原圖寬度動態換算字級
    font      = _load_font(font_size)                    # 載入中文字型

    # ── 頂部：總數量標籤 ─────────────────────────────────────────────────────
    count_label = f"總數量: {len(polygons)}"
    count_bbox  = draw.textbbox((0, 0), count_label, font=font)
    count_tw    = count_bbox[2] - count_bbox[0]
    count_th    = count_bbox[3] - count_bbox[1] + 10

    draw.rectangle(
        [0, 0, count_tw + 6, count_th + 6],
        fill=(0, 0, 0),          # 黑底
        outline=(255, 255, 255), # 白色外框，確保各種背景下都清楚
        width=config.LABEL_BORDER_WIDTH,
    )
    draw.text(
        (3, 3),
        count_label,
        fill=(255, 255, 255),    # 白色文字
        font=font,
    )

    for pts, color, label in zip(polygons, colors, labels):
        pts_int           = pts.astype(np.int32)
        x, y, bw, bh      = cv2.boundingRect(pts_int)  # 取 mask 外接矩形座標

        # color 是 BGR（給 OpenCV 用），PIL 需要 RGB，需交換 B 和 R
        rgb_color = (color[2], color[1], color[0])  # BGR → RGB

        # 量測文字佔用的寬高（multiline 會計算所有行）
        bbox = draw.textbbox((0, 0), label, font=font)
        tw   = bbox[2] - bbox[0]   # 文字區塊寬度（像素）
        th   = bbox[3] - bbox[1] + 10  # 文字區塊高度（像素，含所有行）

        # 文字標籤位置：水平對齊 mask 左緣，垂直貼在 mask 上方
        tx = x                    # 水平：對齊 mask 左邊緣
        ty = max(0, y - th - 6)   # 垂直：mask 頂端往上 th+6px，max(0) 防超出頂部

        # 文字框：與 mask 同色填滿（不透明），黑色外框
        draw.rectangle(
            [tx, ty + th, tx + tw + 6, ty + 2 * th + 6],  # 矩形座標
            fill=rgb_color,            # 與 mask 同色填滿（RGB，不透明）
            outline=(0, 0, 0),         # 黑色外框
            width=config.LABEL_BORDER_WIDTH,
        )

        # 文字：黑色，在彩色背景上對比清晰
        draw.text(
            (tx + 3, ty + th + 3),
            label,
            fill=(0, 0, 0),   # 黑色文字
            font=font,
        )

    return pil_img


def enhance_contrast_pillow(image: Image.Image, factor: float = 2.0) -> Image.Image:
    """
    提高圖片對比度。

    Args:
        image: PIL.Image
        factor:
            1.0 = 原始
            >1.0 = 更高對比
            <1.0 = 更低對比

    Returns:
        增強後的 PIL.Image
    """
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def pil_to_bytes(image: Image.Image, fmt: str = "JPEG") -> bytes:
    """
    將 PIL Image 轉為指定格式的 bytes（用於 API 直接回傳圖片串流）。

    Args:
        image: PIL Image（任意模式）
        fmt:   編碼格式，預設 JPEG

    Returns:
        圖片的原始 bytes
    """
    buf = io.BytesIO()                          # 建立記憶體緩衝區
    image.save(buf, format=fmt, quality=95)     # quality=95：高畫質但比 PNG 小
    return buf.getvalue()                       # 取出緩衝區中的所有 bytes


# ── 內部輔助函式 ──────────────────────────────────────────────────────────────

def _random_colors(n: int) -> list[tuple]:
    """
    產生 n 個互不重複的隨機顏色（BGR 格式）。
    亮度限制在 80-255，避免顏色太暗難以辨識。
    """
    colors: list[tuple] = []   # 已選顏色的列表
    used: set[tuple]   = set() # 已用顏色的集合（避免重複）
    while len(colors) < n:
        c = (
            random.randint(80, 255),  # B 通道
            random.randint(80, 255),  # G 通道
            random.randint(80, 255),  # R 通道
        )
        if c not in used:             # 確保顏色不重複
            used.add(c)
            colors.append(c)
    return colors


def _dynamic_font_size(img_shape: tuple) -> int:
    """
    依原圖寬度動態換算字級，讓標籤大小隨圖片解析度等比縮放。

    換算基準（實測固定比例 width / 125）：
      4000px 寬 → 32pt
      2000px 寬 → 16pt

    Args:
        img_width: 原圖寬度（像素）

    Returns:
        字級（pt），至少為 1，避免極小圖片算出 0 或負值
    """
    return max(16, round(max(img_shape[:2]) / 125))


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """
    載入中文字型，失敗時 fallback 到 PIL 內建預設字型（不支援中文，但不會 crash）。
    """
    try:
        return ImageFont.truetype(config.FONT_PATH, size)  # 嘗試載入 Noto Sans CJK
    except Exception:
        return ImageFont.load_default()  # fallback：無中文但不影響服務啟動
