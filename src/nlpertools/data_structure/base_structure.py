#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji


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

    def build_from_list(self):
        pass

    def __str__(self):
        pass

    @staticmethod
    def pre_order(node):
        stack = []
        res = []
        while stack or node:
            while node:
                res.append(node)
                stack.append(node)
                node = node.left
            node = stack.pop(-1)
            node = node.right
        return res

    def level_order(self, node):
        # 层序遍历
        # 直观觉得递归不行，采用迭代
        # deque表示正在遍历的层
        deque = [node]
        nxt_deque = []
        res = []
        while deque:
            while deque:
                node = deque.pop(0)
                res.append(node.val)
                if node.left:
                    nxt_deque.append(node.left)
                if node.right:
                    nxt_deque.append(node.right)
            deque, nxt_deque = nxt_deque, []
        return res

        pass

    def bfs(self):
        # 具体怎么用迭代写BFS，是根据需求来的。
        # dp 里面放这个吧(node, depth)
        pass

    def mid_order(self, node):
        if node.left:
            self.mid_order(node.left)
        print(node.val)
        if node.right:
            self.mid_order(node.right)

    def post_order(self, node=None):
        pass

    def in_order(self, node):
        # bts的读法
        pass


if __name__ == '__main__':
    a = ListNode([1, 2, 3, 4])
    print(a)
