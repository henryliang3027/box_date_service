"""
usage_report.py — 客戶用量查詢工具
====================================
讀取 data/usage.db（service_api/services/usage.py 寫入），查詢客戶的成功請求紀錄。
同樣的資料也可以在 dashboard 網頁 /admin 查看。

使用方式：
    python3 usage_report.py [--start 起] [--end 迄] [--customer 客戶代號] [--detail] [--csv 路徑]

    --start / --end 接受 YYYY-MM-DD 或 "YYYY-MM-DD HH:MM:SS"，皆為包含；
    只給日期時，--end 包含當天整天。不給則不限制。

範例：
    python3 usage_report.py                                             # 所有客戶的總次數
    python3 usage_report.py --start 2026-10-01 --end 2026-10-31         # 10 月每個客戶的次數
    python3 usage_report.py --customer customer_a --start 2026-10-11 --detail   # 列出每筆：時間、檔名、存檔路徑
    python3 usage_report.py --start 2026-10-01 --end 2026-10-31 --detail --csv oct.csv
"""

import argparse
import csv
import sys
from pathlib import Path

from service_api import config
from service_api.services import usage


def main() -> None:
    parser = argparse.ArgumentParser(description="查詢客戶 API 用量（只含成功的請求）")
    parser.add_argument("--start",    help="起始時間（含）")
    parser.add_argument("--end",      help="結束時間（含）")
    parser.add_argument("--customer", help="只看指定客戶代號")
    parser.add_argument("--detail",   action="store_true", help="列出每一筆請求")
    parser.add_argument("--csv",      help="把結果另存為 CSV")
    args = parser.parse_args()

    conn = usage.connect_readonly()
    if conn is None:
        sys.exit(f"找不到用量資料庫：{config.USAGE_DB_PATH}（尚未有任何成功請求）")

    try:
        if args.detail:
            _, rows = usage.query_requests(conn, args.start, args.end, args.customer)
            rows.reverse()  # 命令列由舊到新列出
        else:
            rows = usage.query_by_customer(conn, args.start, args.end, args.customer)
    except ValueError as e:
        sys.exit(str(e))

    if args.detail:
        for r in rows:
            image_path = Path(config.USAGE_IMAGES_DIR) / r["image_path"] if r["image_path"] else "（存檔失敗）"
            print(f"{r['received_at']}  {r['customer']:<14} {r['original_filename']:<28} boxes={r['num_boxes']:<3} {image_path}")
        print(f"\n共 {len(rows)} 筆")
    else:
        print(f"{'客戶':<16}{'請求次數':>8}{'總箱數':>8}   {'第一筆':<20}{'最後一筆':<20}")
        print("-" * 76)
        for r in rows:
            print(f"{r['customer']:<18}{r['requests']:>8}{r['total_boxes']:>9}   {r['first_request']:<20}{r['last_request']:<20}")
        print("-" * 76)
        print(f"{'合計':<16}{sum(r['requests'] for r in rows):>8}{sum(r['total_boxes'] for r in rows):>9}")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:  # utf-8-sig：Excel 開啟中文不亂碼
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["customer"])
            writer.writeheader()
            writer.writerows(rows)
        print(f"已存成 {args.csv}")


if __name__ == "__main__":
    main()
