#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import re


class Pattern:
    # from where
    # hlink
    pattern = re.compile(r"(https?|ftp|file)://[-A-Za-z0-9+&@#/%?=~_|!:,.;]+[-A-Za-z0-9+&@#/%=~_|]")
    # emoji
    pattern_emoji = re.compile(u"[\U00010000-\U0010ffff\\uD800-\\uDBFF\\uDC00-\\uDFFF]")
