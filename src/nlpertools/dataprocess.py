#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import re
from typing import List, Dict

from . import DB_CONFIG_FILE
from .io.file import read_yaml, readtxt_string
from .utils.package import *

import string

main_special_characters = string.punctuation + string.digits + string.whitespace
other_special_characters = (
    "    　    ￼’“”–ー一▬…✦�­£​•€«»°·═"
    "×士＾˘⇓↓↑←→（）§″′´¿−±∈﻿¢ø‚„½¼¾¹²³―⁃，ˌ¸‹›ʺˈʻ¦‐⠀‰‑≤≥‖"
    "◆●■►▼▲▴∆▻¡★☆✱ːº。¯˜¥ɪ≈†上ン：∼⁄・♡✓⊕․．⋅÷１‟；،、¨ाাी्े◦˚"
    "゜ʼ≖ʼ¤ッツシ℃√！【】‿∞➤～πه۩☛₨➩☻๑٪♥ıॽ《‘©﴿٬？▷Г♫∟™ª₪®「—❖"
    "」﴾》"
)


class Pattern:
    """
    >>> pattern_special_char = re.compile("[{}{}]".format(pattern_special_char_x[1:-1], pattern_special_char_u[1:-1]))
        a = "\U000d8be6asdasdas \x00v啊实打实\x00\x00v阿松大\x00"
        res = re.sub(pattern_special_char, "$",a)
    """
    # some from data-prepare

    # emoji
    """
    # 这也是emoji的取法，不知道pattern全不全
    import emoji  # Use version emoji==1.6.1, otherwise it won't have UNICODE_EMOJI
    emoji = list(emoji.UNICODE_EMOJI["en"].keys())
    """
    emoji_pattern = u"[\U00010000-\U0010ffff\\uD800-\\uDBFF\\uDC00-\\uDFFF]"

    # 特殊的乱码或不可见字符
    # \x 09:\t 0a:\n 0d:\r
    special_char_x_pattern = "[\x00-\x08\x0b\x0c\x0e\x0f\x10-\x19\x1a-\x1f]"
    # 统计大规模语料出来的非正常字符
    special_char_u_pattern = "[\u3000\U000d8be6\U000e0062\U000e0063\U000e0067\U000e0073\U000e0074\U000e007f]"
    special_char_pattern = "{}{}".format(special_char_x_pattern[1:-1], special_char_u_pattern[1:-1])
    non_printing_characters_pattern = f"[{''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))}]"

    # 必须从头匹配，否则无意义的
    # 中文人名
    chinese_name_pattern = "(?:[\u4e00-\u9fa5·]{2,3})"
    # 英文人名
    english_name_pattern = "(^[a-zA-Z][a-zA-Z\s]{0,20}[a-zA-Z]$)"
    # 纯数字
    pure_num_pattern = "\d+"
    # xxxx图/表 之类的表述
    pic_table_descript_pattern = ".{1,15}图"

    # 无需从头匹配的。
    # hlink
    hlink_pattern = r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"
    http_pattern = "(http|https):\/\/([\w.]+\/?)\S*/\S*"
    # 邮箱
    email_pattern = "[A-Za-z0-9\u4e00-\u9fa5]+@[a-zA-Z0-9_-]+(\.[a-zA-Z0-9_-]+)+"
    # html 可能过于严格了
    html_pattern = "<[\s\S]*?>"
    # 重复 “asdasdasdasd”
    repeat_pattern = "(.)\1+"
    # 日期
    day_time_pattern = "\d{1,4}(-)(1[0-2]|0?[1-9])\1(0?[1-9]|[1-2]\d|30|31)"
    # 小时
    hour_time_pattern = "(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d"
    # 股票
    stock_pattern = (
        "(s[hz]|S[HZ])(000[\d]{3}|002[\d]{3}|300[\d]{3}|600[\d]{3}|60[\d]{4})"
    )

    # 一般是需要替换的
    # 多余空格 => " "
    redundancy_space_pattern = " +"
    # 一般用不到 多余换行符号 => " "
    linebreak_pattern = "[\r\n\t]+"


class DataProcess(object):
    """
    数据处理类
    """

    def __init__(self, id_type, patterns_for_filter: List = None, patterns_for_replace: Dict = None):
        """
        pattern_list:
        """
        self.patterns_for_filter = patterns_for_filter
        self.patterns_for_replace = patterns_for_replace

    def process(self, text):
        # 进来的数据都要做的标准化
        text = self.full2half(text)
        # text = self.filter_http(text)
        text = self.filter_html(text)
        text = self.filter_html_special(text)
        # 根据类型与语言分别处理
        text = self.filter_exclusive(text)
        # text = self.trandition2simple(text)
        # text = self.remove_stopwords(text)
        return text

    def filter_whitelist(self, text):
        whitelist = re.compile(
            "[^\u4e00-\u9fa5^0-9a-zA-Z^-^《^》^<^>^【^】^（^）^{^}^–^…^”^“^,^.^;^?^:^‘^~^`^，^。^？^；^！^：^、^·^!^@^#^$^%^&^(^)^|]"
        )
        text = whitelist.sub("", text)
        return text

    def text_split(self, text, language):
        if language == "en":
            text = text[:256]
        elif language == "zh":
            text = text[:510]
        return text

    def trandition2simple(self, text):
        # 仅对中文
        """
        https://juejin.cn/post/7234554420163100728
        """
        text = zhconv.convert('我幹什麼不干你事。', 'zh-cn')
        return text

    def remove_stopwords(self, text):
        new_tokens = []
        if self.language == "en":
            tokens = text.split(" ")
        else:
            tokens = jieba.lcut(text)

        for i in tokens:
            if i in self.stopwords:
                pass
            else:
                new_tokens.append(i)

        return new_tokens

    @staticmethod
    def split_sentence(sentence, language="chinese"):
        """
        分句，英文有nltk，中文怎么能没有好的分句工具呢
        :param sentence:
        :param language:
        :return:
        """
        # sentences->Str
        # example '12“345。”“6789”'
        assert language in ["chinese", "english"], "unsupportable for other language"
        if language == "chinese":
            split_signs = list("。！？…\t")
            other_sign = "”"
        elif language == "english":
            split_signs = list(".!?")
            other_sign = '"'
        else:
            split_signs = list(".!?")
            other_sign = '"'
        sentences = []
        start_idx = 0
        for idx, char in enumerate(sentence):
            if idx == len(sentence) - 1:
                if char in split_signs:
                    sentences.append(sentence[start_idx: idx + 1].strip())
                    start_idx = idx + 1
                else:
                    sentences.append(sentence[start_idx:].strip())
            else:
                if char in split_signs:
                    if sentence[idx + 1] == other_sign:
                        if idx < len(sentence) - 2:
                            # 处理。”。
                            if sentence[idx + 2] not in split_signs:
                                sentences.append(sentence[start_idx: idx + 2].strip())
                                start_idx = idx + 2
                    elif sentence[idx + 1] not in split_signs:
                        sentences.append(sentence[start_idx: idx + 1].strip())
                        start_idx = idx + 1
        sentences = [i.strip() for i in sentences if i.strip()]
        return sentences

    def cut_word(self, text, language):
        if language == "en":
            tokens = text.split(" ")
        else:
            tokens = jieba.lcut(text)
        return tokens

    def full2half(self, text):
        """
        全角转化为半角
        :param text:
        :return:
        """
        ret_str = ""
        for i in text:
            if ord(i) >= 33 + 65248 and ord(i) <= 126 + 65248:
                ret_str += chr(ord(i) - 65248)
            else:
                ret_str += i
        return ret_str

    def filter_html(self, text):
        # 这个比较严格
        """
        过滤html标签
        :param text:
        :return:
        """
        patterns = [
            re.compile("//<![CDATA[[^>]*//]]>", re.I),  # 匹配CDATA
            re.compile("<s*script[^>]*>[^<]*<s*/s*scripts*>", re.I),  # Script
            re.compile("<s*style[^>]*>[^<]*<s*/s*styles*>", re.I),  # style
            re.compile("<brs*?/?>"),  # 处理换行
            re.compile("</?w+[^>]*>"),  # HTML标签
            re.compile("<!--[^>]*-->"),  # HTML注释
        ]
        for pattern in patterns:
            text = pattern.sub("", text)
        return text

    def filter_html_special(self, text):
        """
        替换所有html转义字符
        这个好像只有新闻有？
        :param text:
        :return:
        """
        # TODO html标签应该是 &nbsp 这种，\xa0也是吗
        CHAR_ENTITIES = {
            "&nbsp": " ",
            "160": " ",
            "lt": "<",
            "60": "<",
            "gt": ">",
            "62": ">",
            "amp": "&",
            "38": "&",
            "quot": '"',
            "34": '"',
            "ldquo": '"',
            "rdquo": '"',
            "mdash": "",
            "\xa0": "",
        }

        re_charEntity = re.compile(r"&#?(?P<name>\w+);", re.S)
        sz = re.search(re_charEntity, text)
        while sz:
            entity = sz.group()  # entity全称，如>
            key = sz.group("name")  # 去除&;后entity,如>为gt
            try:
                htmlstr = re_charEntity.sub(CHAR_ENTITIES[key], text, 1)
                text = htmlstr
                sz = re.search(re_charEntity, htmlstr)
            except KeyError:
                # 以空串代替
                htmlstr = re_charEntity.sub("", text, 1)
                text = htmlstr
                sz = re_charEntity.search(htmlstr)
        return text

    def filter_exclusive(self, text):
        """
        去除 @、 #、 表情等twitter、微博“特有”的情况
        :return:
        """
        if self.idType == "selfmedia" and self.language == "zh":
            pattern = r"([\s]\w+(的微博视频)|#|【|】|转发微博)"
            p = re.compile(pattern, re.S)
            text = p.sub("", text)

            dr = re.compile("@\w+", re.S)
            text = dr.sub("", text)

        return text

    def filter_html_tag(self, text):
        # res_tr = r'<a (.*?)></a>'
        # m_tr = re.findall(res_tr,text,re.S|re.M)
        res = re.sub(r"<a.*?>", "", text)
        res = re.sub(r"</a>", "", res)
        res = re.sub(r"<span.*?>", "", res)
        res = re.sub(r"</span>", "", res)
        res = re.sub(r"<img.*?>", "", res)
        res = re.sub(r"<br.*?>", "", res)
        res = re.sub(r"//", "", res)
        res = re.sub(r"@", "", res)
        res = re.sub(r"</", "", res)
        # res = re.sub(r',', '', res)
        # res = re.sub(r'&nbsp;', '', res)
        return res

    @staticmethod
    def uniform_whitespace(
            document,
            whitespace=[
                " ",
                " ",
                " ",
                " ",
                " ",
                "　",
                " ",
                " ",
                " ",
                " ",
                "￼",
                "",
            ],
    ):
        # from https://github.com/bigscience-workshop/data-preparation
        """There are different whitespace characters."""
        whitespace = set(whitespace)
        document = "".join(
            [char if char not in whitespace else " " for char in document]
        )
        return document

    def filter_pattern(self, text):
        """
        返回True表示命中规则，需要过滤
        """
        for pattern in self.patterns_for_filter:
            if re.match(pattern, text):
                return True
        return False

    def replace_pattern(self, text):
        for pattern, replace in self.patterns_for_replace:
            text = re.sub(pattern, replace, text)
        return text


if __name__ == '__main__':
    pattern_for_filter = [Pattern.redundancy_space_pattern,
                          Pattern.repeat_pattern,
                          Pattern.special_char_pattern]
    pattern_for_replace = {
        Pattern.special_char_pattern, " "
    }
    dp = DataProcess(patterns_for_filter=pattern_for_filter, patterns_for_replace=pattern_for_replace)
