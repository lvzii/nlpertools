#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
from typing import List


class ListNode:
    def __init__(self, x):
        self.length = 1
        if type(x) is int:
            self.val = x
            self.next = None
        else:
            # 初始化list，感觉用递归会比较好
            pre = ListNode(x.pop(0))
            head = pre
            while x:
                self.length += 1
                pre.next = ListNode(x.pop(0))
                pre = pre.next
            self.val = head.val
            self.next = head.next

    def add(self):
        pass

    def __str__(self):
        # TODO 循环链表标记出来
        print_string = [self.val]
        tmp = self.next
        # 防止循环链表
        recurrent_num = 0
        while tmp and recurrent_num <= self.length + 10:
            recurrent_num += 1
            print_string.append(tmp.val)
            tmp = tmp.next
        return str(print_string)


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        if type(val) is list:
            pass
        else:
            self.val = val
            self.left = left
            self.right = right

    def __str__(self):
        pass


if __name__ == '__main__':
    a = ListNode([1, 2, 3, 4])
    print(a)
