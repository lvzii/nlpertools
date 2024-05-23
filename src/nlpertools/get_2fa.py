import sys

import pyotp


def get_2af_value(key):
    totp = pyotp.TOTP(key)
    print(totp.now())


if __name__ == '__main__':
    if len(sys.argv) > 1:  # 检查是否有参数传递进来
        key = sys.argv[1]  # 第一个参数是脚本名称，第二个参数是传递进来的key
        print(key)
        get_2af_value(key)
    else:
        print("Please provide a key as an argument.")
