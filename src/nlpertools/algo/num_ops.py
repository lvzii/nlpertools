#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
def gcd(a, b):
    a, b = b, a % b
    if b == 0:
        return a
    return gcd(a, b)
