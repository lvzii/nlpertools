#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import math
from collections import Counter, defaultdict
from itertools import chain

from .. import writetxt_w_list, LoadFromJson, readtxt_list_all_strip
from ..utils.package import *


class TermFinder(object):
    def __init__(self, book_name):
        self.data = {}  # read_data()
        self.book_name = book_name
        self.threshold = 3
        self.bone_words_path = "data/词汇底表.xlsx"
        self.tfidf_path = "tfidf.json"
        self.extracted_path = f"output/terms_{book_name}.txt"
        self.extracted_in_bone_path = f"output/terms_{book_name}_in_bone.txt"
        self.extracted_not_in_bone_path = f"output/terms_{book_name}_not_in_bone.txt"

        self.bone_words, self.avg_bone_len, self.tfidf = self.prepare()

    def prepare(self):
        bone_words = list(pd.read_excel(self.bone_words_path)["術語"])
        avg_bone_len = sum([len(i) for i in bone_words]) // len(bone_words)

        tfidf = LoadFromJson(self.tfidf_path)

        return bone_words, avg_bone_len, tfidf

    def read_data(self):
        # 读已经识别出来的表
        df = pd.read_csv(f"data/{self.book_name}已识别出的词汇.csv", encoding="gbk")
        existed_terms = list(df["术语"])

        # 读分词表
        tokenized_words = readtxt_list_all_strip(f"data/{self.book_name}分词.txt")
        tokenized_words = [i.strip("\ufeff").split("/") for i in tokenized_words]
        counter = Counter(chain(*tokenized_words))

        # 构建父表
        # 小麦->[冬小麦，秋小麦]
        cover_table = defaultdict(list)
        for word_i in counter.keys():
            for word_j in counter.keys():
                if word_j in word_i and word_i != word_j:
                    cover_table[word_j].append(word_i)
        self.data["counter"] = counter
        self.data["words"] = tokenized_words
        self.data["cover_table"] = cover_table
        self.data["existed_terms"] = existed_terms

    def get_tfidf_score(self, word):
        if word in self.tfidf[self.book_name]:
            tf_score = self.tfidf[self.book_name][word]
        else:
            # 停用词(这里只有标点)也不会是术语
            tf_score = -math.inf
        return tf_score

    def get_wlt_score(self, word):
        return len(word) / self.avg_bone_len

    def get_c_score(self, word):
        counter = self.data["counter"]
        cover_table = self.data["cover_table"]
        counter: Counter
        if word in cover_table and cover_table[word]:
            # 嵌套
            x1 = 0
            for super_word in cover_table[word]:
                x1 += counter[super_word]
            x1 /= len(counter.keys())
            x1 = counter[word] - x1
        else:
            # 不是嵌套
            x1 = counter[word]
        c = math.log2(x1 * len(word))
        return c

    def score(self):
        # 读数据
        self.read_data()
        # 找到所有的词
        # words = self.data["words"]
        # existed_terms = self.data["existed_terms"]
        counter = self.data["counter"]
        candidate_words = counter.keys()
        terms = []
        for word in candidate_words:
            if word:
                score = self.get_c_score(word) + self.get_tfidf_score(word) + self.get_wlt_score(word)
                if score > self.threshold:
                    terms.append(word)
        terms_in_bone = set(self.bone_words).intersection(set(terms))
        terms_not_in_bone = set(terms) - set(self.bone_words)
        writetxt_w_list(sorted(terms), self.extracted_path)
        writetxt_w_list(sorted(terms_in_bone), self.extracted_in_bone_path)
        writetxt_w_list(sorted(terms_not_in_bone), self.extracted_not_in_bone_path)


if __name__ == '__main__':
    finder = TermFinder("测试书本")
    finder.score()
