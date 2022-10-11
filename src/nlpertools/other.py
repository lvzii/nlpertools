#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import os
import re
import string
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

from .io.file import writetxt_w_list, writetxt_a
# import numpy as np
# import psutil
# import pyquery as pq
# import requests
# import torch
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.metrics import precision_recall_fscore_support
# from tqdm import tqdm
# from win32evtlogutil import langid
from .utils.package import *

CHINESE_PUNCTUATION = list('，。；：‘’“”！？《》「」【】<>（）、')
ENGLISH_PUNCTUATION = list(',.;:\'"!?<>()')


def seed_everything():
    # seed everything
    seed = 7777777
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机种子确定
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_np_to_py(res):
    np2py = {
        np.float64: float,
        np.int32: int
    }
    news_dict = {}
    for k, v in res.best_params_.items():
        if type(v) in np2py:
            v = np2py[type(v)](v)
            news_dict[k] = v
    return news_dict


def git_push():
    """
    针对国内提交github经常失败，自动提交
    """
    num = -1
    while 1:
        num += 1
        print("retry num: {}".format(num))
        res = os.system("git push")
        print(str(res))
        if not str(res).startswith("fatal"):
            print("scucess")
            break


def snake_to_camel(s: str) -> str:
    """
    author: u
    将 snake case 转换到 camel case.
    :param s: snake case variable
    :return:
    """
    return s.title().replace("_", "")


def camel_to_snake(s: str) -> str:
    """
    将 camel case 转换到 snake case.
    :param s: camel case variable
    :return:
    """
    return reduce(lambda x, y: x + ('_' if y.isupper() else '') + y, s).lower()


def identify_language(text):
    language = langid.classify(text[:200])[0]
    # print(language)
    if language == 'zh':
        return 'zh'
    elif language == 'en':
        return 'en'
    else:
        return 'other'
    # return 'en'


# other ----------------------------------------------------------------------
# 统计词频
def calc_word_count(list_word, mode, path='tempcount.txt', sort_id=1, is_reverse=True):
    word_count = {}
    for key in list_word:
        if key not in word_count:
            word_count[key] = 1
        else:
            word_count[key] += 1
    word_dict_sort = sorted(word_count.items(), key=lambda x: x[sort_id], reverse=is_reverse)
    if mode == 'w':
        for key in word_dict_sort:
            writetxt_a(str(key[0]) + '\t' + str(key[1]) + '\n', path)
    elif mode == 'p':
        for key in word_dict_sort:
            print(str(key[0]) + '\t' + str(key[1]))
    elif mode == 'u':
        return word_dict_sort


# 字典去重
def dupl_dict(dict_list, key):
    new_dict_list, value_set = [], []
    print('去重中...')
    for i in tqdm(dict_list):
        if i[key] not in value_set:
            new_dict_list.append(i)
            value_set.append(i[key])
    return new_dict_list


def multi_thread_run(_task, data):
    with ThreadPoolExecutor() as executor:
        result = list(tqdm(executor.map(_task, data), total=len(ata)))
    return result


def del_special_char(sentence):
    special_chars = ['\ufeff', '\xa0', '\u3000', '\xa0', '\ue627']
    for i in special_chars:
        sentence = sentence.replace(i, '')
    return sentence


def en_pun_2_zh_pun(sentence):
    # TODO 因为引号的问题，所以我没有写
    for i in ENGLISH_PUNCTUATION:
        pass


def spider(url):
    """

    :param url:
    :return:
    """
    if 'baijiahao' in url:
        content = requests.get(url)
        # print(content.text)
        html = pq.PyQuery(content.text)
        title = html('.index-module_articleTitle_28fPT').text()
        res = html('.index-module_articleWrap_2Zphx').text().rstrip('举报/反馈')
        return '{}\n{}'.format(title, res)


def eda(sentence):
    url = 'http://x.x.x.x:x/eda'
    json_data = dict({"sentence": sentence})
    res = requests.post(url, json=json_data)
    return res.json()['eda']


def find_language(text):
    #  TODO 替换为开源包
    letters = list(string.ascii_letters)
    if len(text) > 50:
        passage = text[:50]
        len_passage = 50
    else:
        len_passage = len(text)
    count = 0
    for c in passage:
        if c in letters:
            count += 1
    if count / len_passage > 0.5:
        return "en"
    else:
        return "not en"


def print_prf(y_true, y_pred, label=None):
    # y_true = [0, 1, 2, 1, 1, 2, 3, 1, 1, 1]
    # y_pred = [0, 1, 2, 1, 1, 2, 3, 1, 1, 1]
    # p, r, f, s = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred)
    # print("p\t{}".format(p))
    # print("r\t{}".format(r))
    # print("f\t{}".format(f))
    # print("s\t{}".format(s))
    result = precision_recall_fscore_support(y_true=y_true, y_pred=y_pred, labels=label)

    for i in range(len(label)):
        res = []
        for k in result:
            res.append('%.5f' % k[i])
        print('{}: {} {} {}'.format(label[i], *res[:3]))


def print_cpu():
    p = psutil.Process()
    # pro_info = p.as_dict(attrs=['pid', 'name', 'username'])
    print(psutil.cpu_count())


def stress_test(func, ipts):
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(func, ipts), total=len(ipts)))
    return results


def get_substring_loc(text, subtext):
    res = re.finditer(
        subtext.replace('\\', '\\\\').replace('?', '\?').replace('(', '\(').replace(')', '\)').replace(']',
                                                                                                       '\]').replace(
            '[', '\[').replace('+', '\+'), text)
    l, r = [i for i in res][0].regs[0]
    return l, r


def tf_idf(corpus, save_path):
    tfidfdict = {}
    vectorizer = CountVectorizer()  # 该类会将文本中的词语转换为词频矩阵，矩阵元素a[i][j] 表示j词在i类文本下的词频
    transformer = TfidfTransformer()  # 该类会统计每个词语的tf-idf权值
    tfidf = transformer.fit_transform(
        vectorizer.fit_transform(corpus))  # 第一个fit_transform是计算tf-idf，第二个fit_transform是将文本转为词频矩阵
    word = vectorizer.get_feature_names()  # 获取词袋模型中的所有词语
    weight = tfidf.toarray()  # 将tf-idf矩阵抽取出来，元素a[i][j]表示j词在i类文本中的tf-idf权重
    for i in range(len(weight)):  # 打印每类文本的tf-idf词语权重，第一个for遍历所有文本，第二个for便利某一类文本下的词语权重
        for j in range(len(word)):
            getword = word[j]
            getvalue = weight[i][j]
            if getvalue != 0:  # 去掉值为0的项
                if getword in tfidfdict:  # 更新全局TFIDF值
                    tfidfdict[getword] += float(getvalue)
                else:
                    tfidfdict.update({getword: getvalue})
    sorted_tfidf = sorted(tfidfdict.items(), key=lambda d: d[1], reverse=True)
    to_write = ['{} {}'.format(i[0], i[1]) for i in sorted_tfidf]
    writetxt_w_list(to_write, save_path, num_lf=1)

# 常用函数参考
# import tensorflow as tf
#
# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     tf.config.experimental.set_memory_growth()
