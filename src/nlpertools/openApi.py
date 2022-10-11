#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import hashlib
import json
import time
import uuid

# import requests
from .utils.package import *


def translate_by_youdao(text):
    YOUDAO_URL = 'https://openapi.youdao.com/api'
    APP_KEY = 'xx'
    APP_SECRET = 'xx'

    def _truncate(q):
        if q is None:
            return None
        size = len(q)
        return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]

    def _do_request(data):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        return requests.post(YOUDAO_URL, data=data, headers=headers)

    def _encrypt(signStr):
        hash_algorithm = hashlib.sha256()
        hash_algorithm.update(signStr.encode('utf-8'))
        return hash_algorithm.hexdigest()

    q = text

    data = {}
    data['from'] = '源语言'
    data['to'] = '目标语言'
    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + _truncate(q) + salt + curtime + APP_SECRET
    sign = _encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    data['vocabId'] = "您的用户词表ID"
    youdao_res = _do_request(data)
    res = youdao_res.json()['translation'][0]
    return res


def get_TexSmart(text):
    # text 可以是list， 也可以是str
    # 如果是分类结果 total['cat_list']['name']
    obj = {"str": text, "options": {"text_cat": {"enable": True}}}
    req_str = json.dumps(obj).encode()

    url = "https://texsmart.qq.com/api"
    r = requests.post(url, data=req_str)
    r.encoding = "utf-8"
    print(r.text)
    total = json.loads(r.text)
    return total
