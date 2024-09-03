#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import itertools
import os
import re
import string
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
import math
import datetime
import difflib
import psutil
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
OTHER_PUNCTUATION = list('!@#$%^&*')


def get_diff_parts(str1, str2):
    # 创建一个 SequenceMatcher 对象
    matcher = difflib.SequenceMatcher(None, str1, str2)

    # 获取差异部分
    diff_parts = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace' or tag == 'delete' or tag == 'insert':
            diff_parts.append((tag, str1[i1:i2], str2[j1:j2]))

    return diff_parts


def run_cmd_with_timeout(cmd, timeout):
    """
    https://juejin.cn/post/7391703459803086848
    """
    process = subprocess.Popen(cmd, shell=True, encoding="utf-8", errors="ignore", stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    res = [None]

    def target():
        try:
            ans = process.communicate()
            res[0] = ans
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate()

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        print(f"Terminating {cmd}")
        process.terminate()
        thread.join()
        print("Terminated successfully")
        return False, f"{cmd} is running over {timeout}s"
    if process.returncode == 0:
        # res[0][0] 是output
        return True, res[0][0]
    else:
        return False, res[0][0]


def print_three_line_table(df):
    # TODO 这里需要添加可以支持excel里变红的功能
    import webbrowser

    # import pandas as pd
    # data = {'from_pc': ['valid_data', 'illegal_char', 'more_data'],
    #         'rom_pc': ['another_valid_data', 'illegal_char', 'data']}
    # df = pd.DataFrame(data)

    # 将 DataFrame 转换为 HTML 表格
    html_table = df.to_html(index=False)
    html_table = html_table.replace('border="1"', 'border="0"')

    first_line_px = str(2)
    second_line_px = str(1)
    third_line_px = str(2)
    # 定义三线表的 CSS 样式
    # // thead 表头
    # // tr 行
    # // td 单元格
    head = """<!DOCTYPE html>
    <html lang="zh">
    <head>
        <meta charset="UTF-8">
        <title>页面标题</title>
    </head>"""
    style = """
    <style>

        table {
            border-collapse: collapse;
        }

        tr, td, th {
            text-align: center; /* 水平居中文本 */
            vertical-align: middle; /* 垂直居中文本 */
        }
        thead tr {
            border-top: (first_line_px)px solid black;
            border-bottom: (second_line_px)px solid black;
        }

        thead th {
            border-bottom: (second_line_px)px solid black;
        }

        tbody tr td {
            border-bottom: 0px solid black;
        }

        tbody tr:last-child td {
            border-bottom: (third_line_px)px solid black;
        }
    </style>"""
    style = style.replace("(first_line_px)", first_line_px).replace("(second_line_px)", second_line_px).replace(
        "(third_line_px)", third_line_px)
    # 将 CSS 样式和 HTML 表格结合起来
    html = f"{style}{html_table}"
    print(html)
    temp_file_path = "temp.html"
    # 将 HTML 保存到文件中
    with open(temp_file_path, "w") as f:
        f.write(html)
    webbrowser.open('file://' + os.path.realpath(temp_file_path))


def jprint(obj, depth=0):
    if isinstance(obj, dict):
        sep = "-" * (10 - depth * 3)
        for k, v in obj.items():
            print(depth * "|", sep, k, sep)
            jprint(v)
    elif isinstance(obj, list):
        for v in obj:
            jprint(v, depth + 1)
    else:
        print(obj)


def print_split(sign="=", num=20):
    print(sign * num)


def seed_everything():
    import torch
    # seed everything
    seed = 7777777
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU随机种子确定
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sent_email(mail_user, mail_pass, receiver, title, content, attach_path=None):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.application import MIMEApplication

    mail_host = 'smtp.qq.com'
    mail_user = mail_user
    mail_pass = mail_pass
    sender = mail_user

    message = MIMEMultipart()
    message.attach(MIMEText(content, 'plain', 'utf-8'))
    if attach_path:
        attachment = MIMEApplication(open(attach_path, 'rb').read())
        attachment["Content-Type"] = 'application/octet-stream'
        attachment.add_header('Content-Dispositon', 'attachment',
                              filename=('utf-8', '', attach_path))  # 注意：此处basename要转换为gbk编码，否则中文会有乱码。
        message.attach(attachment)
    message['Subject'] = title
    message['From'] = sender
    message['To'] = receiver

    try:
        smtp_obj = smtplib.SMTP()
        smtp_obj.connect(mail_host, 25)
        smtp_obj.login(mail_user, mail_pass)
        smtp_obj.sendmail(sender, receiver, message.as_string())
        smtp_obj.quit()
        print('send email success')
    except smtplib.SMTPException as e:
        print('send failed', e)


def convert_np_to_py(obj):
    if isinstance(obj, dict):
        return {k: convert_np_to_py(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_to_py(v) for v in obj]
    elif isinstance(obj, np.float64) or isinstance(obj, np.float32):
        return float(obj)
    else:
        return obj


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
        result = list(tqdm(executor.map(_task, data), total=len(data)))
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
    url = 'https://x.x.x.x:x/eda'
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


def squeeze_list(high_dim_list):
    return list(itertools.chain.from_iterable(high_dim_list))


def unsqueeze_list(flatten_list, each_element_len):
    # 该函数是错的，被split_list替代了
    two_dim_list = [flatten_list[i * each_element_len:(i + 1) * each_element_len] for i in
                    range(len(flatten_list) // each_element_len)]
    return two_dim_list

def split_list(input_list, chunk_size):
    # 使用列表推导式将列表分割成二维数组
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]

def auto_close():
    """
    针对企业微信15分钟会显示离开的机制，假装自己还在上班
    """
    import pyautogui as pg
    import time
    import os
    cmd = 'schtasks /create /tn shut /tr "shutdown -s -f" /sc once /st 23:30'
    os.system(cmd)
    while 1:
        pg.moveTo(970, 17, 2)
        pg.click()
        time.sleep(840)


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


class GaussDecay(object):
    """
    当前只实现了时间的，全部使用默认值
    """

    def __init__(self, origin='2022-08-02', scale='90d', offset='5d', decay=0.5, task="time"):
        self.origin = origin
        self.task = task
        self.scale, self.offset = self.translate(scale, offset)
        self.decay = decay
        self.time_coefficient = 0.6
        self.related_coefficient = 0.4

    def translate(self, scale, offset):
        """
        将领域的输入转化为标准
        :return:
        """
        if self.task == "time":
            scale = 180
            offset = 5
        else:
            scale = 180
            offset = 5
        return scale, offset

    @staticmethod
    def translated_minus(field_value):
        origin = datetime.datetime.now()
        field_value = datetime.datetime.strptime(field_value, '%Y-%m-%d %H:%M:%S')
        return (origin - field_value).days

    def calc_exp(self):
        pass

    def calc_liner(self):
        pass

    def calc_gauss(self, raw_score, field_value):
        """
        $$S(doc)=exp(-\frac{max(0,|fieldvalues_{doc}-origin|-offset)^2}{2σ^2})$$ -
        $$σ^2=-scale^2/(2·ln(decay))$$
        :param raw_score:
        :param field_value:
        :return:
        """
        numerator = max(0, (abs(self.translated_minus(field_value)) - self.offset)) ** 2
        sigma_square = -1 * self.scale ** 2 / (2 * math.log(self.decay, math.e))
        denominator = 2 * sigma_square
        s = math.exp(-1 * numerator / denominator)
        return round(self.time_coefficient * s + self.related_coefficient * raw_score, 7)


if __name__ == '__main__':
    gauss_decay = GaussDecay()
    res = gauss_decay.calc_gauss(raw_score=1, field_value="2021-05-29 14:31:13")
    print(res)
    # res = gauss_decay.calc_gauss(raw_score=1, field_value="2022-05-29 14:31:13")
    # print(res)
    # res = gauss_decay.calc_gauss(raw_score=1, field_value="2022-05-29 14:31:13")
    # print(res)
    # res = gauss_decay.calc_gauss(raw_score=1, field_value="2022-05-29 14:31:13")
    # print(res)

# 常用函数参考
# import tensorflow as tf
#
# gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
# sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
# for gpu in tf.config.experimental.list_physical_devices('GPU'):
#     tf.config.experimental.set_memory_growth()
