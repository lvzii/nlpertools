#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
from collections import defaultdict


# from sortedcontainers import SortedDict, SortedList

# 树状数组只能维护前缀“操作和”(前缀和，前缀积，前缀最大最小)，而线段树可以维护区间操作和。

# 线段树
class SegmentTree:
    """
    https://www.zhihu.com/question/346961479/answer/2274087021
    性质：线段树的每一个树节点其实都存储了一个「区间（段）的信息」
    通过add添加
    """

    pass


# 树状数组（二进制下标树） 模板
class BIT:
    """
    TODO 以前在logseq写过笔记，整理到web上
    代码来自https://leetcode.cn/problems/number-of-recent-calls/solutions/1472043/by-ac_oier-evqe/下的评论
    """

    def __init__(self, n: int):
        self.size = n
        self.tree = defaultdict(int)

    @staticmethod
    def _lowbit(index: int) -> int:
        # TODO 同样整理到web
        return index & -index

    def add(self, index: int, delta: int) -> None:
        """
        delta为index位置加的值
        """
        while index <= self.size:
            self.tree[index] += delta
            index += self._lowbit(index)

    def query(self, index: int) -> int:
        if index > self.size:
            index = self.size
        res = 0
        while index > 0:
            res += self.tree[index]
            index -= self._lowbit(index)
        return res

    def sumRange(self, left: int, right: int) -> int:
        return self.query(right) - self.query(left - 1)


class BITUsageDemo:
    """
    查找区间值的出现次数
    """

    def __init__(self):
        self.bit = BIT(10)

    def add(self, x: int):
        self.bit.add(x, 1)

    def query(self, x, y):
        return self.bit.sumRange(x, y)


class Trie:
    def __init__(self):
        self.children = [None] * 26
        self.isEnd = False

    def insert(self, word: str) -> None:
        node = self
        for ch in word:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                node.children[ch] = Trie()
            node = node.children[ch]
        node.isEnd = True

    def search_prefix(self, prefix: str):
        node = self
        for ch in prefix:
            ch = ord(ch) - ord("a")
            if not node.children[ch]:
                return None
            node = node.children[ch]

        return node

    def search(self, word: str) -> bool:
        node = self.search_prefix(word)
        return node is not None and node.isEnd

    def starts_with(self, prefix: str) -> bool:
        return self.search_prefix(prefix) is not None


if __name__ == "__main__":
    bit_usage_demo = BITUsageDemo()
    bit_usage_demo.add(1)
    bit_usage_demo.add(2)
    res = bit_usage_demo.query(1, 2)
    print(res)
    # 前缀树
    # obj = Trie()
    # obj.insert(word)
    # param_2 = obj.search(word)
    # param_3 = obj.startsWith(prefix)
