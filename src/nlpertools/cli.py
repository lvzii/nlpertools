import argparse
import os
import uuid
import sys

import pyotp

"""
如何Debug cli.py
"""


def git_push():
    """
    针对国内提交github经常失败，自动提交
    """
    num = -1
    while 1:
        num += 1
        print("retry num: {}".format(num))
        info = os.system("git push --set-upstream origin main")
        print(str(info))
        if not str(info).startswith("fatal"):
            print("scucess")
            break


def git_pull():
    """
    针对国内提交github经常失败，自动提交
    """
    num = -1
    while 1:
        num += 1
        print("retry num: {}".format(num))
        info = os.system("git pull")
        print(str(info))
        if not str(info).startswith("fatal") and not str(info).startswith("error"):
            print("scucess")
            break


def get_mac_address():
    mac = uuid.UUID(int=uuid.getnode()).hex[-12:]
    mac_address = ":".join([mac[e:e + 2] for e in range(0, 11, 2)])
    print("mac address 不一定准确")
    print(mac_address)
    return mac_address


def get_2af_value(key):
    """
    key应该是7位的
    """
    print(key)
    totp = pyotp.TOTP(key)
    print(totp.now())


def main():
    parser = argparse.ArgumentParser(description="CLI tool for git operations and getting MAC address.")
    parser.add_argument('--gitpush', action='store_true', help='Perform git push operation.')
    parser.add_argument('--gitpull', action='store_true', help='Perform git push operation.')
    parser.add_argument('--mac_address', action='store_true', help='Get the MAC address.')

    parser.add_argument('--get_2fa', action='store_true', help='Get the 2fa value.')
    parser.add_argument('--get_2fa_key', type=str, help='Get the 2fa value.')

    args = parser.parse_args()

    if args.gitpush:
        git_push()
    elif args.gitpull:
        git_pull()
    elif args.mac_address:
        get_mac_address()
    elif args.get_2fa:
        if args.get_2fa_key:
            get_2af_value(args.get_2fa_key)
        else:
            print("Please provide a key as an argument.")
    else:
        print("No operation specified. Use --gitpush or --get_mac_address.")


if __name__ == '__main__':
    main()
