"""
services/box_matcher.py — 紙箱品名資料庫模糊比對
==================================================
Gemini 對整張圖辨識出的 product_name 是原始 OCR 文字（可能包含錯字、
用詞略有差異、品牌與口味順序不一致等）。本模組將該文字與
`box_name.json`（brand / name / keywords）資料庫做模糊比對，
取得該紙箱正確、統一的 brand / name。

比對邏輯（與 API_Test/box_matcher.py 相同）：
  1. 先用每筆資料的 keywords 做子字串比對（去除空白後比對），
     命中越多、越長的關鍵字分數越高。
  2. 若沒有任何關鍵字命中，改用整體字串相似度（difflib）做模糊比對，
     相似度須達到 FUZZY_MATCH_THRESHOLD 才採用。
  3. 若都無法比對到資料庫中的紙箱，代表資料庫中沒有建立此紙箱，
     回傳 {"brand": None, "name": <原始 product_name>}。

使用方式：
    from service_api.services.box_matcher import match_product
    matched = match_product(gemini_product_name)
    # matched: {"brand": "雁牌", "name": "薑母茶"}
"""

import difflib
import json

from service_api import config

FUZZY_MATCH_THRESHOLD = 0.6

_db_cache: list[dict] | None = None


def _normalize(text: str | None) -> str:
    return (text or "").replace(" ", "").replace("　", "")


def load_db(db_path: str | None = None) -> list[dict]:
    """載入 box_name.json 的 products 清單（未指定路徑時使用 config.BOX_DB_PATH，並快取）。"""
    global _db_cache
    if db_path is None:
        if _db_cache is not None:
            return _db_cache
        with open(config.BOX_DB_PATH, encoding="utf-8") as f:
            data = json.load(f)
        _db_cache = data["products"]
        return _db_cache

    with open(db_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["products"]


def _keyword_score(entry: dict, norm_text: str) -> int:
    score = 0
    for keyword in entry.get("keywords", []):
        norm_keyword = _normalize(keyword)
        if norm_keyword and norm_keyword in norm_text:
            score += len(norm_keyword)
    return score


def _fuzzy_score(entry: dict, norm_text: str) -> float:
    candidates = [_normalize(entry.get("brand", "")) + _normalize(entry.get("name", ""))]
    candidates += [_normalize(k) for k in entry.get("keywords", [])]
    best_ratio = 0.0
    for candidate in candidates:
        if not candidate:
            continue
        ratio = difflib.SequenceMatcher(None, candidate, norm_text).ratio()
        best_ratio = max(best_ratio, ratio)
    return best_ratio


def match_product(product_name: str | None, db: list[dict] | None = None) -> dict:
    """依照 box_name.json 資料庫，對 Gemini 辨識出的 product_name 做模糊比對。

    Returns:
        {"brand": str | None, "name": str}
        比對到資料庫中的紙箱時，brand/name 採用資料庫的正確值；
        比對不到時，brand 為 None，name 沿用原始 product_name。
    """
    if db is None:
        db = load_db()

    norm_text = _normalize(product_name)
    if not norm_text:
        return {"brand": None, "name": product_name}

    best_entry = None
    best_score = 0
    for entry in db:
        score = _keyword_score(entry, norm_text)
        if score > best_score:
            best_score = score
            best_entry = entry

    if best_entry is not None:
        return {"brand": best_entry["brand"], "name": best_entry["name"]}

    best_entry = None
    best_ratio = 0.0
    for entry in db:
        ratio = _fuzzy_score(entry, norm_text)
        if ratio > best_ratio:
            best_ratio = ratio
            best_entry = entry

    if best_entry is not None and best_ratio >= FUZZY_MATCH_THRESHOLD:
        return {"brand": best_entry["brand"], "name": best_entry["name"]}

    return {"brand": None, "name": product_name}
