#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
from ..io.file import readtxt_list_all_strip


def find_sentence_covered_vocab(vocab, sentences):
    """
    找到词典中
    此为参照写法，具体用的时候复制出去用避免重复构建
    """

    from ahocorasick import Automaton
    atm = Automaton()
    for word in vocab:
        atm.add_word(word, word)
    atm.make_automaton()

