#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import codecs
import json
import pickle
import random
import time
from itertools import (takewhile, repeat)
import pandas as pd
# import omegaconf
# import yaml
from ..utils.package import *

LARGE_FILE_THRESHOLD = 1e5


def read_yaml(path, omega=False):
    if omega:
        return omegaconf.OmegaConf.load(path)
    return yaml.load(codecs.open(path), Loader=yaml.FullLoader)


def _merge_file(filelist, save_filename, shuffle=False):
    contents = []
    for file in filelist:
        content = readtxt_list_all_strip(file)
        contents.extend(content)
    if shuffle:
        random.shuffle(contents)
    writetxt_w_list(contents, save_filename)


# file's io ----------------------------------------------------------------------
def iter_count(file_name):
    """
    最快的文件行数统计，不知道和wc -l 谁快
    author: unknown
    """
    buffer = 1024 * 1024
    with codecs.open(file_name, 'r', 'utf-8') as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)


# 需要加入进度条的函数包括
"""
readtxt_list_all_strip
save_to_json
load_from_json
"""


# 读txt文件 一次全读完 返回list 去换行
def readtxt_list_all_strip(path, encoding='utf-8'):
    file_line_num = iter_count(path)
    lines = []
    with codecs.open(path, 'r', encoding) as r:
        if file_line_num > LARGE_FILE_THRESHOLD:
            iter_obj = tqdm(enumerate(r.readlines()), total=file_line_num)
        else:
            iter_obj = enumerate(r.readlines())

        for ldx, line in iter_obj:
            lines.append(line.strip('\n').strip("\r"))
        return lines


# 读txt 一次读一行 最后返回list
def readtxt_list_each(path):
    lines = []
    with codecs.open(path, 'r', 'utf-8') as r:
        line = r.readline()
        while line:
            lines.append(line)
            line = r.readline()
    return lines


def readtxt_list_each_strip(path):
    """
    yield方法
    """
    with codecs.open(path, 'r', 'utf-8') as r:
        line = r.readline()
        while line:
            yield line.strip("\n").strip("\r")
            line = r.readline()


# 读txt文件 一次全读完 返回list
def readtxt_list_all(path):
    with codecs.open(path, 'r', 'utf-8') as r:
        lines = r.readlines()
        return lines


# 读byte文件 读成一条string
def readtxt_byte(path, encoding="utf-8"):
    with codecs.open(path, 'rb') as r:
        lines = r.read()
        lines = lines.decode(encoding)
        return lines.replace('\r', '')


# 读txt文件 读成一条string
def readtxt_string(path, encoding="utf-8"):
    with codecs.open(path, 'r', encoding) as r:
        lines = r.read()
        return lines.replace('\r', '')


# 写txt文件覆盖
def writetxt_w(txt, path, r='w'):
    with codecs.open(path, r, 'utf-8') as w:
        w.writelines(txt)


# 写txt文件追加
def writetxt_a(txt, path):
    with codecs.open(path, 'a', 'utf-8') as w:
        w.writelines(txt)


def writetxt(txt, path, encoding="utf-8"):
    with codecs.open(path, 'w', encoding) as w:
        w.write(txt)


def writetxt_wb(txt, path):
    with codecs.open(path, 'wb') as w:
        w.write(txt)


# 写list 覆盖
def writetxt_w_list(list, path, num_lf=1):
    with codecs.open(path, 'w', "utf-8") as w:
        for i in list:
            w.write(i)
            w.write("\n" * num_lf)


# 写list 追加
def writetxt_a_list(list, path, num_lf=2):
    with codecs.open(path, 'a', "utf-8") as w:
        for i in list:
            w.write(i)
            w.write("\n" * num_lf)


def save_to_json(content, path):
    with codecs.open(path, "w", "utf-8") as w:
        json.dump(content, w, ensure_ascii=False, indent=1)


def load_from_json(path):
    with codecs.open(path, "r", "utf-8") as r:
        content = json.load(r)
        return content


# 读txt文件 读成一条string if gb2312
def readtxt_string_all_encoding(path):
    try:
        with codecs.open(path, 'rb', "utf-8-sig") as r:
            lines = r.read()
            return lines
    except:
        try:
            with codecs.open(path, 'rb', "utf-8") as r:
                lines = r.reacd()
                return lines
        except:
            try:
                with codecs.open(path, 'rb', "big5") as r:
                    lines = r.read()
                    return lines
            except:
                print(path)
                with codecs.open(path, 'rb', "gb2312", errors='ignore') as r:
                    lines = r.read()
                    return lines


def readtxt_list_all_encoding(path):
    try:
        with codecs.open(path, 'rb', "utf-8-sig") as r:
            lines = r.readlines()
            return lines
    except:
        try:
            with codecs.open(path, 'rb', "utf-8") as r:
                lines = r.readlines()
                return lines
        except:
            try:
                with codecs.open(path, 'rb', "big5") as r:
                    lines = r.readlines()
                    return lines
            except:
                with codecs.open(path, 'rb', "gb2312", errors='ignore') as r:
                    lines = r.readlines()
                    return lines


# line by line
def save_to_jsonl(corpus, path):
    with open(path, 'w', encoding='utf-8') as wt:
        for i in corpus:
            wt.write(json.dumps(i, ensure_ascii=False))
            wt.write('\n')


# line by line
def load_from_jsonl(path):
    file_line_num = iter_count(path)
    if file_line_num > 1e5:
        with open(path, 'r', encoding='utf-8') as rd:
            corpus = []
            while True:
                line = rd.readline()
                if line:
                    corpus.append(json.loads(line))
                else:
                    break
        return corpus
    else:
        with open(path, 'r', encoding='utf-8') as rd:
            corpus = []
            while True:
                line = rd.readline()
                if line:
                    corpus.append(json.loads(line))
                else:
                    break
        return corpus


def pickle_save(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def pickle_load(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_to_csv(df, save_path, index_flag=False):
    with open(save_path, 'wb+') as csvfile:
        csvfile.write(codecs.BOM_UTF8)
    df.to_csv(save_path, mode='a', index=index_flag)


def save_to_mongo():
    # fake
    """
    示例

    """
    pass

def load_from_mongo():
    pass


def unmerge_cells_df(df) -> pd.DataFrame:
    for column in df.columns:
        values = []
        for i in df[column]:
            if pd.isna(i):
                values.append(values[-1])
            else:
                values.append(i)
        df[column] = values
    return df