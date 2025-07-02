import argparse
import os
import uuid
import sys
from .dataprocess import startwith


def run_git_command(command):
    """
    循环执行git命令，直到成功
    """
    print(command)
    num = -1
    while True:
        num += 1
        print(f"retry num: {num}")
        info = os.system(command)
        print(str(info))
        # 检查命令执行结果，若未出现错误则认为执行成功
        if not startwith(str(info), ["fatal", "error", "128", "1"]):
            print("success")
            print(f"success info : ##{info}##")
            break


def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    mac_address = ":".join([mac[e:e + 2] for e in range(0, 11, 2)])
    print("mac address 不一定准确")
    print(mac_address)
    return mac_address


def get_2af_value(key):
    import pyotp
    """
    key应该是7位的
    """
    print(key)
    totp = pyotp.TOTP(key)
    print(totp.now())


def start_gpu_usage_notify_server():
    from flask import Flask

    app = Flask(__name__)

    @app.route("/notify", methods=["GET"])
    def notify():
        # 这里可以根据需要动态生成通知内容
        usage = os.popen("nvidia-smi --query-gpu=memory.used --format=csv").read().split("\n")[1:]
        res = 0
        for edx, each in enumerate(usage):
            if each.startswith("0"):
                res += 1
        print(res)
        return str(res), 200

    app.run(host="0.0.0.0", port=5000)


def start_gpu_usage_notify_client():
    import requests
    from plyer import notification
    import time

    SERVER_URL = 'http://127.0.0.1:5000/notify'  # 服务器的 API 地址

    def notify(text):
        # 使用 plyer 发送通知
        notification.notify(
            title='远程通知',
            message=text,
            timeout=10  # 10秒的通知显示时间
        )

    """定时轮询服务器获取通知"""
    while True:
        try:
            response = requests.get(SERVER_URL)
            if response.status_code == 200:
                num = int(response.text)
                if num > 0:
                    notify(f"服务器有{num}张卡")
                print(f"服务器有{num}张卡")
            else:
                print("服务器没有新通知")
        except Exception as e:
            print(f"与服务器连接失败: {e}")

        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="CLI tool for git operations and other functions.")
    parser.add_argument('git_command', nargs='*', help='Any git command (e.g., push, pull)')
    parser.add_argument('--mac_address', action='store_true', help='Get the MAC address.')
    parser.add_argument('--get_2fa', action='store_true', help='Get the 2fa value.')
    parser.add_argument('--get_2fa_key', type=str, help='Get the 2fa value.')
    parser.add_argument('--monitor_gpu_cli', action='store_true', help='monitor gpu cli')
    parser.add_argument('--monitor_gpu_ser', action='store_true', help='monitor gpu ser')

    args = parser.parse_args()

    if args.git_command:
        git_cmd = " ".join(args.git_command)
        run_git_command(git_cmd)
    elif args.mac_address:
        get_mac_address()
    elif args.monitor_gpu_cli:
        start_gpu_usage_notify_client()
    elif args.monitor_gpu_ser:
        start_gpu_usage_notify_server()
    elif args.get_2fa:
        if args.get_2fa_key:
            get_2af_value(args.get_2fa_key)
        else:
            print("Please provide a key as an argument.")
    else:
        print("No operation specified.")


if __name__ == '__main__':
    main()