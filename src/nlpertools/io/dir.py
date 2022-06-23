#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import codecs
import os
import json
import pickle
import time
# dir ----------------------------------------------------------------------
def j_mkdir(name):
    os.makedirs(name, exist_ok=True)


def get_filename(path):
    '''
    返回路径最后的文件名
    :param path:
    :return:
    '''
    # path = r'***/**/***.txt'
    filename = os.path.split(path)[-1]
    return filename


# TODO 还没写
def walk():
    paths = os.walk(r'F:\**\**\**\***')
    for root, dir, files in paths:
        for name in files:
            if name == '***.**':
                # os.remove(os.path.join(root, name))
                yield


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

