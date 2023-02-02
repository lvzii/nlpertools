#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
from ..utils.package import sns, plt


def confused_matrix(confuse_matrix):
    sns.set()
    f, ax = plt.subplots()
    ticklabels = ["l1", "l2", "l31"]
    sns.heatmap(confuse_matrix, annot=True, fmt=".3g", ax=ax, cmap='rainbow',
                xticklabels=ticklabels, yticklabels=ticklabels)  # 画热力图

    ax.set_title('confusion matrix')  # 标题
    ax.set_xlabel('predict')  # x轴
    ax.set_ylabel('true')  # y轴
    plt.show()

    f.savefig('tmp.jpg', bbox_inches='tight')