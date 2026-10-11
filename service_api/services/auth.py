"""
services/auth.py — 客戶 API Key 驗證
======================================
從 api_keys.json 讀取 { key: 客戶代號 } 對照表，驗證請求 header 的 X-API-Key。

使用方式（路由參數）：
    customer: str = Security(require_customer)

新增客戶：在 api_keys.json 加一行，重啟 uvicorn 生效。
產生 key：python -c "import secrets; print('bdk_' + secrets.token_urlsafe(32))"
"""

import json                                       # 讀取 api_keys.json

from fastapi import HTTPException, Security       # 401 回應、Swagger Authorize 依賴注入
from fastapi.security import APIKeyHeader         # 讓 Swagger UI 出現 Authorize 按鈕

from service_api import config                    # API_KEYS_PATH


def _load_api_keys(path: str) -> dict[str, str]:
    """
    讀取 { key: 客戶代號 } 對照表。
    檔案不存在或格式錯誤時回傳空 dict（所有請求都會被拒絕，fail closed）。
    """
    try:
        with open(path, encoding="utf-8") as f:
            keys = json.load(f)
    except FileNotFoundError:
        print(f"⚠️ [auth] 找不到 {path}，所有需驗證的請求都會回 401")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️ [auth] {path} 格式錯誤（{e}），所有需驗證的請求都會回 401")
        return {}
    print(f"🔑 [auth] 已載入 {len(keys)} 把 API Key（客戶：{sorted(set(keys.values()))}）")
    return keys


# 服務啟動時載入一次；修改 api_keys.json 後需重啟 uvicorn
API_KEYS: dict[str, str] = _load_api_keys(config.API_KEYS_PATH)

# auto_error=False：缺少 header 時由 require_customer 自行回 401 + 中文訊息（預設會回 403）
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_customer(api_key: str | None = Security(_api_key_header)) -> str:
    """驗證 X-API-Key，通過時回傳客戶代號，否則回 401。"""
    if not api_key:
        raise HTTPException(status_code=401, detail="缺少 X-API-Key")
    customer = API_KEYS.get(api_key)
    if customer is None:
        raise HTTPException(status_code=401, detail="無效的 API Key")
    return customer
