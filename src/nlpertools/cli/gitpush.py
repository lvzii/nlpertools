import os


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


if __name__ == '__main__':
    git_push()
