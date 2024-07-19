import argparse
import os
import uuid


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
    print(mac_address)
    return mac_address


def main():
    parser = argparse.ArgumentParser(description="CLI tool for git operations and getting MAC address.")
    parser.add_argument('--gitpush', action='store_true', help='Perform git push operation.')
    parser.add_argument('--gitpull', action='store_true', help='Perform git push operation.')
    parser.add_argument('--mac_address', action='store_true', help='Get the MAC address.')

    args = parser.parse_args()

    if args.gitpush:
        git_push()
    elif args.gitpull:
        git_pull()
    elif args.mac_address:
        get_mac_address()
    else:
        print("No operation specified. Use --gitpush or --get_mac_address.")


if __name__ == '__main__':
    main()
