#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import re
import string
from typing import List

import numpy as np

# from . import DB_CONFIG_FILE # cannot import name 'DB_CONFIG_FILE' from partially initialized module 'nlpertools'
from .utils.package import *

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
    emoji_pattern = "[\U00010000-\U0010ffff\\uD800-\\uDBFF\\uDC00-\\uDFFF]"

    # 特殊的乱码或不可见字符
    # \x 09:\t 0a:\n 0d:\r
    special_char_x_pattern = "[\x00-\x08\x0b\x0c\x0e\x0f\x10-\x19\x1a-\x1f]"
    # 统计大规模语料出来的非正常字符
    special_char_u_pattern = (
        "[\u3000\U000d8be6\U000e0062\U000e0063\U000e0067\U000e0073\U000e0074\U000e007f]"
    )
    special_char_pattern = "{}{}".format(
        special_char_x_pattern[1:-1], special_char_u_pattern[1:-1]
    )
    non_printing_characters_pattern = (
        f"[{''.join(map(chr, list(range(0, 32)) + list(range(127, 160))))}]"
    )

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
    hlink_pattern = (
        r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]"
    )
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

    # 微博视频等
    weibo_pattern = r"([\s]\w+(的微博视频)|#|【|】|转发微博)"
    # @
    at_pattern = "@\w+"

    # from https://github.com/bigscience-workshop/data-preparation pii
    year_patterns = [
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/][1-2][0-9]{3})(?:$|[\s@,?!;:\'\"(.\p{Han}])",
        # yyyy-yyyy or yyyy/yyyy
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}[\p{Pd}/.][0-3][0-9][\p{Pd}/.][0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])",
        # yyyy-mm-dd or yyyy-dd-mm or yyyy/mm/dd or yyyy/dd/mm or yyyy.mm.dd or yyyy.dd.mm
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/.][0-3][0-9][\p{Pd}/.](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])",
        # mm-dd-yyyy or dd-mm-yyyy or mm/dd/yyyy or dd/mm/yyyy or mm.dd.yyyy or dd.mm.yyyy or the same but with yy instead of yyyy
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([0-3][0-9][\p{Pd}/](?:[0-9]{2}|[1-2][0-9]{3}))(?:$|[\s@,?!;:\'\"(.\p{Han}])",
        # mm-yyyy or mm/yyyy or the same but with yy
        r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([1-2][0-9]{3}-[0-3][0-9])(?:$|[\s@,?!;:\'\"(.\p{Han}])",
        # yyyy-mm or yyyy/mm
    ]

    # Patterns for high-risk character strings
    id_pattern = r'(?:^|[\b\s@?,!;:\'\")(.\p{Han}])([A-Za-z]*(?:[\p{Pd}]*\p{Nd}){6,})(?:$|[\b\s@?,!;:\'\")(.\p{Han}])'
    # https://regex101.com/r/JQkmh8/2
    # key_pattern = r'(?:^|[\b\s@?,!;:\'\")(.\p{Han}])((?:(?:[A-Za-z]+[\p{Nd}\p{Pd}\/\+\=:]+|[\p{Nd}\p{Pd}\/\+\=:]+[A-Za-z]+)){4,}|(?:(?:\p{Nd}{3,}|[A-Z]+\p{Nd}+[A-Z]*|\p{Nd}+[A-Z]+\p{Nd}*)[\s\p{Pd}]?){4,})(?:$|[\b\s\p{Han}@?,!;:\'\"])'
    # https://regex101.com/r/JQkmh8/5
    key_pattern = r'(?:^|[\b\s@?,!:;\'\")(.\p{Han}])((?:(?:[A-Za-z]+[\p{Nd}\p{Pd}\/\+\=:_]+|[\p{Nd}\p{Pd}\/\+\=:]+[A-Za-z]+)){4,}|(?:(?:\p{Nd}{3,}|[A-Z]+\p{Nd}+[A-Z]*|\p{Nd}+[A-Z]+\p{Nd}*)[ \p{Pd}]?){3,})(?:$|[\b\s\p{Han}@?,!;:\'\")(.])'
    ipv4_pattern = r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?:\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)){3}'
    ipv6_pattern = r'(?:[0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,7}:|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:(?:(?::[0-9a-fA-F]{1,4}){1,6})|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(?:ffff(?::0{1,4}){0,1}:){0,1}(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(?:2[0-4]|1{0,1}[0-9]){0,1}[0-9])'
    ip_pattern = r"(?:^|[\b\s@?,!;:\'\")(.\p{Han}])(" + r"|".join(
        [ipv4_pattern, ipv6_pattern]) + ")(?:$|[\s@,?!;:\'\"(.\p{Han}])"

    # https://regex101.com/r/EpA5B7/1
    email_line_pattern = r'''
        (?<= ^ | [\b\s@,?!;:)('".\p{Han}<] )
        (
        [^\b\s@?!;,:)('"<]+
        @
        [^\b\s@!?;,/]*
        [^\b\s@?!;,/:)('">.]
        \.
        \p{L} \w{1,}
        )
        (?= $ | [\b\s@,?!;:)('".\p{Han}>] )
    '''

    # https://regex101.com/r/mOqi1s/3
    # user_pattern = r'(?:^|[\s@,?!;:\'\")(\p{Han}])(@[^\s@,?!;:\'\")(]{3,})'
    user_pattern = r'''
    (?<= ^ | [)(\s@,?!;:'"\p{Han}] )
    (@
        [^)(\s@,?!;:'"]{3,}
    )
    '''


class CalcPPL(object):
    # ppl计算
    # https://www.scribendi.ai/comparing-bert-and-gpt-2-as-language-models-to-score-the-grammatical-correctness-of-a-sentence/
    def __init__(self, model_type, model_path, tokenizer_path):
        self.model_type = model_type
        self.model, self.tokenizer = self._init_model(model_type, model_path, tokenizer_path)

    @staticmethod
    def _init_model(model_type, model_path, tokenizer_path):
        if model_type == "ngram":
            model = kenlm.Model(model_path)
            tokenizer = sentencepiece.SentencePieceProcessor()
            tokenizer.load(tokenizer_path)
        elif model_type == "bert":
            model = BertForMaskedLM.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path)
        elif model_type == "gpt":
            model = GPT2LMHeadModel.from_pretrained(model_path)
            tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        else:
            model = tokenizer = None
            assert "model_type should in ngram bert gpt"
        return model, tokenizer

    def ppl(self, sentence):
        # 根据model_type自动选择
        if self.model_type == "ngram":
            return self.ppl_ngram(sentence)
        elif self.model_type == "ngram":
            return self.ppl_bert(sentence)
        else:
            return self.ppl3_gpt(sentence)

    def ppl_ngram(self, sentence):
        pass

    def ppl_bert_2(self, sentence):
        # 忘记哪来的
        tokenizer = self.tokenizer
        model = self.tokenizer
        tokenize_input = tokenizer.tokenize(sentence)
        tokenize_input = tokenize_input
        tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
        with torch.no_grad():
            loss = model(tensor_input, labels=tensor_input)[0]
        return np.exp(loss.detach().numpy())

    # [1] Salazar J, Liang D, Nguyen T Q, et al. Masked Language Model Scoring[C]//Proceedings of ACL. 2020: 2699-2712.
    def ppl_bert(self, sentence):
        tokenizer = self.tokenizer
        model = self.tokenizer
        with torch.no_grad():
            tokenize_input = tokenizer.tokenize(sentence)
            tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
            sen_len = len(tokenize_input)
            sentence_loss = 0.

            for i, word in enumerate(tokenize_input):
                # add mask to i-th character of the sentence
                tokenize_input[i] = '[MASK]'
                mask_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])

                output = model(mask_input)

                prediction_scores = output[0]
                softmax = nn.Softmax(dim=0)
                ps = softmax(prediction_scores[0, i]).log()
                word_loss = ps[tensor_input[0, i]]
                sentence_loss += word_loss.item()

                tokenize_input[i] = word
            ppl = np.exp(-sentence_loss / sen_len)
            # print("困惑度：", ppl)
            return ppl

    def ppl3_gpt(self, text):
        from torch.nn import CrossEntropyLoss
        # 这里用 GPT2LMHeadModel
        inputs = self.tokenizer([text], padding='max_length', max_length=50, truncation=True, return_tensors="pt")
        bs, sl = inputs['input_ids'].size()
        outputs = self.model(**inputs, labels=inputs['input_ids'])
        logits = outputs[1]
        # Shift so that tokens < n predict n
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = inputs['input_ids'][:, 1:].contiguous()
        shift_attentions = inputs['attention_mask'][:, 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss(ignore_index=0, reduction="none")
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).detach().reshape(bs, -1)
        meanloss = loss.sum(1) / shift_attentions.sum(1)
        ppl = torch.exp(meanloss).numpy().tolist()
        return ppl[0]

    def test(self):
        sentence = "输入句子："
        ppl = self.ppl_bert_2(sentence)
        ppl2 = self.ppl_bert(sentence)
        print(ppl)
        print(ppl2)


class TextProcess(object):
    """
    数据处理类
    这是基类，如果是定制化的语言处理，请继承该类
    """

    def __init__(
            self,
            patterns_filter: List = None,
            patterns_replace: List[List] = None,
            words_filter: List = []
    ):
        """
        pattern_list:
        """
        self.patterns_filter, self.patterns_replace = self._pre_compile_pattern(
            patterns_filter, patterns_replace
        )
        self.words_filter = words_filter

    @staticmethod
    def _pre_compile_pattern(patterns_filter, patterns_replace):
        complied_patterns_replace, complied_patterns_filter = [], []
        for i in patterns_filter:
            complied_patterns_filter.append(re.compile(i))
        for i in patterns_replace:
            complied_patterns_replace.append((re.compile(i[0]), i[1]))
        return complied_patterns_filter, complied_patterns_replace

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

    def filter_words(self, text):
        # 根据词典，命中返回True，需要过滤掉

        for word in self.words_filter:
            if word in text:
                return True
        return False

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
        text = zhconv.convert("我幹什麼不干你事。", "zh-cn")
        return text

    def remove_stopwords(self, text):
        import jieba

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
        import jieba

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
        for pattern in self.patterns_filter:
            if re.match(pattern, text):
                return True
        return False

    def replace_pattern(self, text):
        for pattern, replace in self.patterns_replace:
            text = re.sub(pattern, replace, text)
        return text

    def calc_proportion_zh(self,text):
        text = text.strip()
        # 如果是中国英文的情况，并且英文有空格分开
        if " " in text:
            pass
        chinese_count = 0
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                chinese_count += 1
            else:
                pass
class CopyFunc():
    # from https://github.com/lemon234071/clean-dialog
    def is_chinese_char(cp):
        """Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        return (
                (cp >= 0x4E00 and cp <= 0x9FFF)
                or (cp >= 0x3400 and cp <= 0x4DBF)  #
                or (cp >= 0x20000 and cp <= 0x2A6DF)  #
                or (cp >= 0x2A700 and cp <= 0x2B73F)  #
                or (cp >= 0x2B740 and cp <= 0x2B81F)  #
                or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
                or (cp >= 0xF900 and cp <= 0xFAFF)
                or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
        )

    def contains_Chinese(seq):
        for char in seq:
            cp = ord(char)
            if is_chinese_char(cp):
                return True
        return False


class EnTextProcess(object):
    pass


def convert2markdown(table: list) -> str:
    df = pd.DataFrame(table[1:], columns=table[0])

    return df.to_markdown(index=False)


def convert_fullwidth2_basic(sentence):
    # 参照：https://fuhaoku.net/U+FF21
    new_sentence = ""
    for char in sentence:
        if 65281 <= ord(char) <= 65374:
            char = chr(ord(char) - 65248)
        new_sentence += char
    return new_sentence


def convert_basic2fullwidth(sentence):
    new_sentence = ""
    for char in sentence:
        if 33 <= ord(char) <= 126:
            char = chr(ord(char) + 65248)
        new_sentence += char
    return new_sentence

if __name__ == "__main__":
    pattern_for_filter = [
        Pattern.redundancy_space_pattern,
        Pattern.repeat_pattern,
        Pattern.special_char_pattern,
    ]
    pattern_for_replace = [(Pattern.special_char_pattern, " ")]

    dp = TextProcess(
        patterns_filter=pattern_for_filter, patterns_replace=pattern_for_replace
    )
    dp.process(text="demo")
