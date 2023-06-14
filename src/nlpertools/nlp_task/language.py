#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import os

from ..utils.package import *


def identify_language(text):
    # 这个方法不好，已经被模型取代了
    language = langid.classify(text[:200])[0]
    # print(language)
    if language == 'zh':
        return 'zh'
    elif language == 'en':
        return 'en'
    else:
        return 'other'
    # return 'en'


class LanguageClassify():
    import fasttext

    # 用fasttext的模型
    def __init__(self):
        self.lid_model = self.get_lid_model()

    @staticmethod
    def get_lid_model():
        path_fasttext_model = "lid.176.ftz"
        assert os.path.exists(
            path_fasttext_model), f"请从下面连接获取语言模型，放在当前目录下{os.path.abspath(__file__)}https://fasttext.cc/docs/en/language-identification.html"
        model_lang_id = fasttext.load_model(path_fasttext_model)
        return model_lang_id

    def identify_language_2(self, text):
        language = self.lid_model.predict(text)[0][0].lstrip("__label__")
        return language
