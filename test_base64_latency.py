"""
test_base64_latency.py — 測量 POST /api/v1/detect/image_base64 的分段 latency
====================================================================

跟 test_latency.py 測的是同一組三段 latency，差別只在於這個端點的
請求格式是 JSON body（image_base64 欄位），不是 multipart/form-data 上傳檔案：

  1. upload   : client 送出 → server 收到（上行網路傳輸）
  2. process  : server 收到 → 辨識完成（server 端運算，YOLO + Gemini）
  3. download : server 回傳 → client 收到（下行網路傳輸）

server（service_api/main.py 的 TimingMiddleware）會在每個 response 附上：
  - X-Server-Recv-Time            server 收到 request 的 epoch time
  - X-Server-Send-Time            server 送出 response 的 epoch time
  - X-Server-Inference-Done-Time  辨識完成的 epoch time（僅 /detect/image_base64 會有）

由於 client 與 server 可能是不同機器、時鐘不同步，若直接拿
「server 時間戳」與「client 時間戳」相減會有時鐘偏移（clock skew）誤差。
所以本script 會先用類似 NTP 的方式，對 /health 端點打多輪 request，
估計 client 與 server 之間的時鐘偏移量（offset），再用這個 offset
校正 upload / download 兩段的計算。

  offset（server 時間 - client 時間的推估值）：
    對每一輪：
      t0 = client 送出時間
      t3 = client 收到時間
      server_recv / server_send 從 response header 取得
      offset_sample = ((server_recv - t0) + (server_send - t3)) / 2
    取多輪中 RTT（t3 - t0）最小的幾筆 offset 的中位數，
    RTT 越小代表上下行越對稱、offset 估計越準（NTP 演算法的簡化版）。

server 端運算耗時（process，第2段）完全不受時鐘偏移影響，
因為 X-Server-Recv-Time 與 X-Server-Inference-Done-Time 都是同一台機器的時間戳。

⚠️ 這個 offset 校正假設上下行網路延遲對稱，實際網路不一定對稱，
   估計值可能仍有誤差；若條件允許，建議搭配兩台機器都做好 NTP 校時
   一起使用，結果會更可靠。

用法：
    python test_base64_latency.py --url http://<server-ip>:8080 --image sample_images/xxx.jpg
    python test_base64_latency.py --url http://127.0.0.1:8080 --image sample_images/xxx.jpg --rounds 10

參數：
    --url          server base url（不含 path），例如 http://192.168.1.10:8080
    --image        要上傳測試的圖片路徑
    --rounds       /detect/image_base64 測試次數（預設 5）
    --sync-rounds  時鐘校正用的 /health 測試次數（預設 10）
    --csv          若指定路徑，會把每輪結果存成 CSV
"""

from __future__ import annotations

import argparse
import base64
import csv
import statistics
import sys
import time
from pathlib import Path

import requests


def measure_clock_offset(base_url: str, rounds: int) -> float:
    """
    用類似 NTP 的方式估計 (server_time - client_time) 的偏移量（秒）。
    對 /health 打多輪 request，取 RTT 最小的幾筆 offset 的中位數。
    """
    samples: list[tuple[float, float]] = []  # (rtt, offset)
    for _ in range(rounds):
        t0 = time.time()
        resp = requests.get(f"{base_url}/health", timeout=10)
        t3 = time.time()
        resp.raise_for_status()

        server_recv = float(resp.headers["X-Server-Recv-Time"])
        server_send = float(resp.headers["X-Server-Send-Time"])

        rtt = t3 - t0
        offset = ((server_recv - t0) + (server_send - t3)) / 2.0
        samples.append((rtt, offset))

    # 取 RTT 最小的一半樣本，RTT 越小代表上下行越對稱，offset 估計越準
    samples.sort(key=lambda s: s[0])
    best = samples[: max(1, len(samples) // 2)]
    offset = statistics.median(o for _, o in best)
    return offset


def measure_detect_image_base64(base_url: str, image_base64: str, offset: float) -> dict:
    """打一次 /api/v1/detect/image_base64，回傳拆分後的三段 latency（單位：秒）。"""
    payload = {"image_base64": image_base64, "include_image": False}

    t0 = time.time()  # client 送出時間
    resp = requests.post(
        f"{base_url}/api/v1/detect/image_base64",
        json=payload,
        timeout=120,
    )
    t3 = time.time()  # client 收到時間
    resp.raise_for_status()

    server_recv = float(resp.headers["X-Server-Recv-Time"])
    server_send = float(resp.headers["X-Server-Send-Time"])
    server_done = float(resp.headers["X-Server-Inference-Done-Time"])

    # 校正到同一個時鐘座標系（用 server 時鐘為基準）：client 時間 + offset ≈ server 時鐘下的時間
    t0_corrected = t0 + offset
    t3_corrected = t3 + offset

    upload_s   = server_recv - t0_corrected          # 第1段：client 送出 → server 收到
    process_s  = server_done - server_recv            # 第2段：server 收到 → 辨識完成（不受時鐘偏移影響）
    download_s = t3_corrected - server_send            # 第3段：server 回傳 → client 收到
    total_s    = t3 - t0                                # 總 latency（client 端量測，不受 offset 影響）

    return {
        "total_ms":    total_s * 1000,
        "upload_ms":   upload_s * 1000,
        "process_ms":  process_s * 1000,
        "download_ms": download_s * 1000,
        "status_code": resp.status_code,
    }


def main():
    parser = argparse.ArgumentParser(description="測量 /api/v1/detect/image_base64 的分段 latency")
    parser.add_argument("--url", required=True, help="server base url，例如 http://192.168.1.10:8080")
    parser.add_argument("--image", required=True, help="要上傳測試的圖片路徑")
    parser.add_argument("--rounds", type=int, default=5, help="/detect/image_base64 測試次數（預設 5）")
    parser.add_argument("--sync-rounds", type=int, default=10, help="時鐘校正用的 /health 測試次數（預設 10）")
    parser.add_argument("--csv", default=None, help="若指定路徑，把每輪結果存成 CSV")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"❌ 找不到圖片：{image_path}", file=sys.stderr)
        sys.exit(1)

    print(f"🔧 正在校正 client/server 時鐘偏移（{args.sync_rounds} 輪 /health）...")
    offset = measure_clock_offset(base_url, args.sync_rounds)
    print(f"   估計偏移量 offset = {offset * 1000:.2f} ms（server 時間 - client 時間）")

    # 圖片只需編碼一次，後面每輪重複使用同一份 base64 字串
    image_base64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

    print(f"\n🚀 開始測試 {base_url}/api/v1/detect/image_base64（共 {args.rounds} 輪，圖片：{image_path.name}）")
    results = []
    for i in range(1, args.rounds + 1):
        r = measure_detect_image_base64(base_url, image_base64, offset)
        results.append(r)
        print(
            f"  [{i}/{args.rounds}] total={r['total_ms']:7.1f} ms  "
            f"upload={r['upload_ms']:7.1f} ms  "
            f"process={r['process_ms']:7.1f} ms  "
            f"download={r['download_ms']:7.1f} ms"
        )

    print("\n📊 統計結果（median / mean / min / max，單位 ms）")
    for key, label in [
        ("total_ms", "total   (client 送出 → client 收到)"),
        ("upload_ms", "upload  (client 送出 → server 收到)"),
        ("process_ms", "process (server 收到 → 辨識完成)"),
        ("download_ms", "download(server 回傳 → client 收到)"),
    ]:
        vals = [r[key] for r in results]
        print(
            f"  {label:38s} median={statistics.median(vals):8.1f}  "
            f"mean={statistics.mean(vals):8.1f}  "
            f"min={min(vals):8.1f}  max={max(vals):8.1f}"
        )

    if args.csv:
        csv_path = Path(args.csv)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "total_ms", "upload_ms", "process_ms", "download_ms", "status_code"])
            writer.writeheader()
            for i, r in enumerate(results, 1):
                writer.writerow({"round": i, **r})
        print(f"\n💾 已存成 CSV：{csv_path}")


if __name__ == "__main__":
    main()
