#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji

def build(pattern_string):
    """
    构建模式串的PMT
    [zhihu](https://www.zhihu.com/question/21923021/answer/281346746)

    """
    # 构建pattern需要回溯的位置，
    backtrace_points = [0] * len(pattern_string)
    main_pointer, pattern_pointer = 0, -1
    backtrace_points[0] = -1
    while main_pointer < len(pattern_string) - 1:
        if pattern_pointer == -1 or pattern_string[pattern_pointer] == pattern_string[main_pointer]:
            main_pointer += 1
            pattern_pointer += 1
            backtrace_points[main_pointer] = pattern_pointer
        else:
            pattern_pointer = backtrace_points[pattern_pointer]
    return backtrace_points


def build_2(needle: str):
    # 这写的比第一种简洁
    # 查找方法也是自己，唯一就是判断结束条件，不是用-1了
    m = len(needle)
    if m == 0:
        return 0

    pmt = [0] * m
    pattern_pointer = 0
    for main_pointer in range(1, m):
        while pattern_pointer > 0 and needle[main_pointer] != needle[pattern_pointer]:
            pattern_pointer = pmt[pattern_pointer - 1]
        if needle[main_pointer] == needle[pattern_pointer]:
            pattern_pointer += 1
        pmt[main_pointer] = pattern_pointer
    return pmt


def find_after_build(main_string, pattern_string):
    backtracker = build(pattern_string)
    # print(backtracker)
    main_pointer, pattern_pointer = -1, -1
    while main_pointer <= len(main_string) - 1:
        if pattern_pointer == -1 or pattern_string[pattern_pointer] == main_string[main_pointer]:
            # 这是返回首次匹配时main的位置
            if pattern_pointer == len(pattern_string) - 1:
                return main_pointer - len(pattern_string) + 1
            pattern_pointer += 1
            main_pointer += 1
        else:
            pattern_pointer = backtracker[pattern_pointer]
    return -1


def find(main_string, pattern_string):
    """
    模式匹配
    一边构建字串的回溯点，一边判断模式是否匹配
    """
    if len(main_string) < len(pattern_string):
        return False
    main_string = " " + main_string
    backtrace_points = [0] * (len(main_string) + 1)
    main_pointer, pattern_pointer = 0, -1
    backtrace_points[0] = -1
    while main_pointer < len(main_string):
        if pattern_pointer == -1 or pattern_string[pattern_pointer] == main_string[main_pointer]:
            if pattern_pointer == len(pattern_string) - 1:
                return True
            main_pointer += 1
            pattern_pointer += 1
            backtrace_points[main_pointer] = pattern_pointer
        else:
            pattern_pointer = backtrace_points[pattern_pointer]
    return False


if __name__ == '__main__':
    test_main_string = "abababc"
    test_pattern_string = "abababc"

    res = build(test_pattern_string)
    print(res)
    res = build_2(test_pattern_string)
    print(res)
    # res = find(test_main_string, test_pattern_string)
    # print(res)
    #
    # res = find_after_build(test_main_string, test_pattern_string)
    # print(res)
