#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
from typing import List


class ListNode:
    def __init__(self, x):
        if type(x) is int:
            self.val = x
            self.next = None
        else:
            # 初始化list，感觉用递归会比较好
            pre = ListNode(x.pop(0))
            head = pre
            while x:
                pre.next = ListNode(x.pop(0))
                pre = pre.next
            self.val = head.val
            self.next = head.next

    def add(self):
        pass

    def __str__(self):
        print_string = [self.val]
        tmp = self.next
        while tmp:
            print_string.append(tmp.val)
            tmp = tmp.next
        return str(print_string)


a = ListNode([1, 2, 3, 4])
print(a)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
