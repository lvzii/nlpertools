#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
try:
    import pandas as pd
except:
    pass


def remind_assert():
    assert "train_extension" in ["csv", "json"], "`train_file` should be a csv or a json file."


def remind_dir():
    reminder = "os.path.dirname(os.path.abspath(__file__))"


def remind_me():
    reminder = '''
- 数据获取
- 数据清洗
 - dataclean
- 预标注
 - get_TexSmart
- 校对
- 添加数据训练
- 评价
- 纠正标注数据'''
    print(reminder)


class PandasLookup():
    def merge_data(a, b):
        a = pd.DataFrame({
            "esid": [1, 2, 1]
        })
        b = pd.DataFrame({
            "esid": [2, 3, 1],
            "content": ['z', 'v', 'b'],
            "other": [1, 2, 3]
        })
        c = pd.DataFrame({
            "esid": [1, 2],
            "content": ["b", 'z']
        })
        res = a.merge(b, left_on='esid', right_on='esid', how='left')
        return res


"""

from flask import send_file, send_from_directory
import os

@app.route("/download/<filename>", methods=['GET'])
def download_file(filename):
    # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
    directory = os.getcwd()  # 假设在当前目录
    return send_from_directory(directory, filename, as_attachment=True)
"""
