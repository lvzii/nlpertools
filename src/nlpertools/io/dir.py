#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import os
from pathlib import Path


# dir ----------------------------------------------------------------------
def j_mkdir(name):
    os.makedirs(name, exist_ok=True)


def j_walk(name, suffix=None):
    paths = []
    for root, dirs, files in os.walk(name):
        for file in files:
            path = os.path.join(root, file)
            if not (suffix and not path.endswith(suffix)):
                paths.append(path)
    return paths


def windows_to_wsl_path(windows_path):
    # 转换驱动器号
    if windows_path[1:3] == ':\\':
        drive_letter = windows_path[0].lower()
        path = windows_path[2:].replace('\\', '/')
        wsl_path = f'/mnt/{drive_letter}{path}'
    else:
        # 如果路径不是以驱动器号开头，则直接替换路径分隔符
        wsl_path = windows_path.replace('\\', '/').replace("'", "\'")

    return wsl_path


def get_filename(path, suffix=True) -> str:
    """
    返回路径最后的文件名
    :param path:
    :return:
    """
    # path = r'***/**/***.txt'
    filename = os.path.split(path)[-1]
    if not suffix:
        filename = filename.split('.')[0]
    return filename


def j_listdir(dir_name, including_dir=True):
    filenames = os.listdir(dir_name)
    if including_dir:
        return [os.path.join(dir_name, filename) for filename in filenames]
    else:
        return list(filenames)


def j_listdir_yield(dir_name, including_dir=True):
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
