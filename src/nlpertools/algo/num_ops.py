#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
def gcd(a, b):
    """
    math.gcd()等包对于gcd的实现源码中看不到
    实现方法；辗转相除法
    """
    a, b = b, a % b
    if b == 0:
        return a
    return gcd(a, b)
