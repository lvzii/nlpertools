#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
from ..utils.package import plt


def confused_matrix(confuse_matrix):
    import seaborn as sns
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


def plot_histogram(data, bin_size):
    """
    画直方图，超过1000的统一按1000算
    :param data:
    :param bin_size:
    :return:
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import MaxNLocator
    # 将超过1000的值改为1000
    def process_lengths(data):
        return [length if length <= 1000 else 1003 for length in data]

    # 前闭后开
    min_num, max_num = 0, 1000
    # min_num, max_num = min(data), max(data)

    plt.figure(figsize=(12, 8))
    processed_data = process_lengths(data)
    bins = np.arange(0, 1000 + 2 * bin_size, bin_size)
    # 绘制直方图
    n, new_bins, patches = plt.hist(processed_data, bins=bins, edgecolor='black', color='skyblue', alpha=0.7,
                                    linewidth=0)

    # 添加"∞"的标签
    # bins会改变
    plt.gca().set_xticks(bins)
    plt.gca().set_xticklabels([str(i) for i in plt.xticks()[0][:-1]] + ["∞"])

    mean_val = np.mean(data)
    plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1)
    plt.text(mean_val + bin_size / 10, max(n) * 0.9, f'Mean: {mean_val:.2f}', color='red')

    # 添加标题和标签
    plt.title('Module Line Number Distribution', fontsize=16, fontweight='bold')
    plt.xlabel('module line number', fontsize=14)
    plt.ylabel('frequency', fontsize=14)

    # 添加网格
    plt.grid(True, linestyle='--', alpha=0.6)

    # 美化x轴和y轴的刻度
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # 在每个柱状图上显示数值
    for i in range(len(patches)):
        plt.text(patches[i].get_x() + patches[i].get_width() / 2, patches[i].get_height(),
                 str(int(n[i])), ha='center', va='bottom', fontsize=12)
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    # 显示图表
    plt.show()


if __name__ == '__main__':
    # 调整区间大小
    bin_size = 50
    # 示例模块长度数据
    plot_histogram([1, 100, 999, 1000, 1002, 1100, 1150], bin_size)
