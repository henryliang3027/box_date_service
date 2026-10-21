"""
manage_web_users.py — 手機 Web 版（/app）帳號管理
==================================================
帳號存在 web_users.json（不進 git），密碼只存雜湊。修改後不用重啟 uvicorn。

使用方式：
    python3 manage_web_users.py add <帳號> <客戶代號>    # 新增帳號（會要求輸入兩次密碼）
    python3 manage_web_users.py passwd <帳號>            # 改密碼（該帳號已登入的手機全部登出）
    python3 manage_web_users.py remove <帳號>            # 停用帳號（已登入的手機立即失效）
    python3 manage_web_users.py list                     # 列出帳號與客戶代號

客戶代號必須是 api_keys.json 已有的 value（例如 customer_a），add 時會檢查，
帳號與 API Key 的用量因此記在同一個客戶名下。
"""

import argparse
import getpass
import json
import os
import sys

from service_api import config
from service_api.services import web_auth

_MIN_PASSWORD_LEN = 8


def _load() -> dict:
    try:
        with open(config.WEB_USERS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def _save(users: dict) -> None:
    # 先寫暫存檔再改名，避免寫到一半時伺服器讀到壞掉的檔案；權限 600 只有自己能讀
    tmp = config.WEB_USERS_PATH + ".tmp"
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)
    os.replace(tmp, config.WEB_USERS_PATH)


def _known_customers() -> set[str]:
    """api_keys.json 裡已設定的客戶代號（只讀 value，不碰 key 本身）。"""
    try:
        with open(config.API_KEYS_PATH, encoding="utf-8") as f:
            return set(json.load(f).values())
    except (OSError, json.JSONDecodeError):
        return set()


def _ask_password() -> str:
    while True:
        pw = getpass.getpass("密碼：")
        if len(pw) < _MIN_PASSWORD_LEN:
            print(f"密碼至少 {_MIN_PASSWORD_LEN} 個字元")
            continue
        if getpass.getpass("再輸入一次：") != pw:
            print("兩次輸入不一致")
            continue
        return pw


def main() -> None:
    parser = argparse.ArgumentParser(description="管理手機 Web 版（/app）帳號")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_add = sub.add_parser("add", help="新增帳號")
    p_add.add_argument("username")
    p_add.add_argument("customer", help="客戶代號（與 api_keys.json 的 value 一致）")
    sub.add_parser("passwd", help="改密碼").add_argument("username")
    sub.add_parser("remove", help="停用帳號").add_argument("username")
    sub.add_parser("list", help="列出帳號")
    args = parser.parse_args()

    users = _load()

    if args.cmd == "list":
        if not users:
            print(f"（{config.WEB_USERS_PATH} 沒有任何帳號）")
        for name, u in sorted(users.items()):
            print(f"{name:<20} {u['customer']}")
        return

    if args.cmd == "add":
        if args.username in users:
            sys.exit(f"帳號 {args.username} 已存在（改密碼請用 passwd）")
        # 客戶代號必須是 api_keys.json 已有的客戶，避免打錯字讓用量記到不存在的客戶名下
        known = _known_customers()
        if args.customer not in known:
            sys.exit(f"api_keys.json 沒有客戶代號 {args.customer}（目前有：{', '.join(sorted(known)) or '無'}）")
        users[args.username] = {"customer": args.customer, "password": web_auth.hash_password(_ask_password())}
        _save(users)
        print(f"已新增 {args.username}（客戶：{args.customer}）")
        return

    if args.username not in users:
        sys.exit(f"找不到帳號 {args.username}")

    if args.cmd == "passwd":
        users[args.username]["password"] = web_auth.hash_password(_ask_password())
        _save(users)
        n = web_auth.delete_user_sessions(args.username)
        print(f"已更新 {args.username} 的密碼，登出 {n} 個裝置")
    elif args.cmd == "remove":
        del users[args.username]
        _save(users)
        n = web_auth.delete_user_sessions(args.username)
        print(f"已停用 {args.username}，登出 {n} 個裝置")


if __name__ == "__main__":
    main()
