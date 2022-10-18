#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
def foo(num):
    return num & -num


def foo2(num):
    return num & (num - 1)
