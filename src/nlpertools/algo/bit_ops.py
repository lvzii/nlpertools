#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
def foo(num):
    return num & -num


def foo2(num):
    """
    raw: 0 1 2 3 4 5 6 7 8 9
    res: 0 0 0 2 0 4 4 6 0 8
    """
    return num & (num - 1)


def _lowbit(index: int) -> int:
    """
    raw: 0 1 2 3 4 5 6 7 8 9
    res: 0 1 2 1 4 1 2 1 8 1
    """
    return index & -index

if __name__ == '__main__':
    for i in range(10):
        print(i, end=" ")
    print()
    for i in range(10):
        print(foo2(i), end=" ")