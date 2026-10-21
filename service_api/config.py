"""
config.py — 所有設定集中管理
=============================
修改設定時只需改這一個檔案。
也可透過環境變數覆蓋（適合 Docker / 不同環境部署）。

用法：
    from service_api import config
    print(config.YOLO_MODEL_PATH)
"""

import os                    # 讀取環境變數
from pathlib import Path     # 跨平台路徑操作

from dotenv import load_dotenv  # 讀取 .env 檔案，載入環境變數

# ── 專案路徑 ──────────────────────────────────────────────────────────────────
# config.py 在 service_api/ 裡，上一層（parent.parent）才是專案根目錄
ROOT_DIR: Path = Path(__file__).parent.parent

# 載入專案根目錄的 .env（GEMINI_API_KEY 等機密設定），不覆蓋已存在的環境變數
load_dotenv(ROOT_DIR / ".env")

# ── YOLO 設定 ─────────────────────────────────────────────────────────────────
# YOLO 模型權重路徑，可用環境變數 YOLO_MODEL_PATH 覆蓋
YOLO_MODEL_PATH: str = os.getenv(
    "YOLO_MODEL_PATH",
    str(ROOT_DIR / "models/yolo_model/box_segmentation/best_26x_seg_20260828.pt"),
)
# 低於此信心值的偵測結果會被過濾（0.0~1.0，越高越嚴格）
YOLO_CONF_THRESHOLD: float = float(os.getenv("YOLO_CONF", "0.70"))
# 推論時將圖片縮放至此邊長（須與訓練時一致）
YOLO_IMG_SIZE: int = int(os.getenv("YOLO_IMGSZ", "640"))

# ── Gemini 設定 ───────────────────────────────────────────────────────────────
# Gemini API Key，需在環境變數 GEMINI_API_KEY 設定
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
# 使用的 Gemini 模型名稱
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

# ── 品名資料庫比對設定 ─────────────────────────────────────────────────────────
# 紙箱品名資料庫路徑（brand/name/keywords），Gemini 辨識出的 product_name 會與此資料庫
# 做模糊比對，取得正確的 brand/name；可用環境變數 BOX_DB_PATH 覆蓋
BOX_DB_PATH: str = os.getenv("BOX_DB_PATH", str(ROOT_DIR / "box_name.json"))

# ── 視覺化設定 ────────────────────────────────────────────────────────────────
# 中文字型路徑（Ubuntu Noto Sans CJK）
FONT_PATH: str = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
# 標籤文字大小依原圖寬度動態換算（見 image_utils._dynamic_font_size），不再用固定值
# mask 填色的透明度（0.0=完全透明，1.0=完全不透明）
MASK_ALPHA: float = 0.4
# 文字標籤外框線寬（像素）
LABEL_BORDER_WIDTH: int = 2

# ── 未知品項 mask 儲存路徑（供人工複核用）────────────────────────────────────
SEGMENTATION_UNKNOWN_DIR: str = os.getenv(
    "SEGMENTATION_UNKNOWN_DIR",
    str(ROOT_DIR / "enrollment/segmentation_unknown"),
)

# ── FastAPI 設定 ──────────────────────────────────────────────────────────────
# API 路徑前綴（所有偵測端點都掛在這個前綴下）
API_PREFIX: str = "/api/v1"

# ── 客戶 API Key 設定 ─────────────────────────────────────────────────────────
# { key: 客戶代號 } 對照表（不進 git），可用環境變數 API_KEYS_PATH 覆蓋
API_KEYS_PATH: str = os.getenv("API_KEYS_PATH", str(ROOT_DIR / "api_keys.json"))

# ── 客戶用量紀錄設定 ──────────────────────────────────────────────────────────
# 用量資料庫（每次成功請求一筆，計費依據），可用環境變數 USAGE_DB_PATH 覆蓋
USAGE_DB_PATH: str = os.getenv("USAGE_DB_PATH", str(ROOT_DIR / "data" / "usage.db"))
# 客戶上傳原圖存放資料夾（依 <客戶>/<YYYYMMDD>/ 分層），可用環境變數 USAGE_IMAGES_DIR 覆蓋
USAGE_IMAGES_DIR: str = os.getenv("USAGE_IMAGES_DIR", str(ROOT_DIR / "data" / "customer_images"))

# ── 用量 dashboard（/admin）登入設定 ──────────────────────────────────────────
# 瀏覽器會跳出帳密視窗（HTTP Basic）；請在 .env 設定。ADMIN_PASSWORD 留空時 dashboard 停用
ADMIN_USER: str = os.getenv("ADMIN_USER", "admin")
ADMIN_PASSWORD: str = os.getenv("ADMIN_PASSWORD", "")

# ── 手機 Web 版（/app）登入設定 ───────────────────────────────────────────────
# 現場人員帳號 { 帳號: {customer, password(雜湊)} }（不進 git），用 manage_web_users.py 管理
WEB_USERS_PATH: str = os.getenv("WEB_USERS_PATH", str(ROOT_DIR / "web_users.json"))
# 登入 session 資料庫（只存 token 的雜湊），可用環境變數 WEB_SESSIONS_DB_PATH 覆蓋
WEB_SESSIONS_DB_PATH: str = os.getenv("WEB_SESSIONS_DB_PATH", str(ROOT_DIR / "data" / "web_sessions.db"))
# 登入後幾小時要重新登入
WEB_SESSION_HOURS: int = int(os.getenv("WEB_SESSION_HOURS", "12"))
# cookie 只在 HTTPS 送出；用 http://<區網 IP> 在手機上測試時設成 false，正式環境務必維持 true
WEB_COOKIE_SECURE: bool = os.getenv("WEB_COOKIE_SECURE", "true").lower() in ("1", "true", "yes")

# ── 除錯設定 ──────────────────────────────────────────────────────────────────
# True 時，/detect/image 會印出圖片接收時間（hh:mm:ss）、
# YOLO 偵測耗時、以及 Gemini 推論耗時
DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")
