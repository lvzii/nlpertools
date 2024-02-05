#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import os
from pathlib import Path


# dir ----------------------------------------------------------------------
def j_mkdir(name):
    os.makedirs(name, exist_ok=True)


def get_filename(path) -> str:
    """
    返回路径最后的文件名
    :param path:
    :return:
    """
    # path = r'***/**/***.txt'
    filename = os.path.split(path)[-1]
    return filename


def j_listdir(dir_name, including_dir=True):
    #  yield
    filenames = os.listdir(dir_name)
    for filename in filenames:
        if including_dir:
            yield os.path.join(dir_name, filename)
        else:
            yield filename


# 合并文件 TODO 还没写
def imgrate_files(path):
    filenames = os.listdir(path)
    return None


def case_sensitive_path_exists(path: str, relative_path=False):
    """
    https://juejin.cn/post/7316725867086692391
    Check if the path exists in a case-sensitive manner.
    """
    # 构造成Path
    if relative_path:
        path = Path.cwd() / path
    else:
        path = Path(path)
    if not path.exists():
        return False
    # resolved_path是系统里的该文件实际名称
    resolved_path = path.resolve()
    return str(resolved_path) == str(path)
