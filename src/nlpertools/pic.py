#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji


def convert(path):
    from pdf2image import convert_from_path

    pages = convert_from_path(path, 500)

    # 保存
    num = 1
    for page in pages:
        page.save('out{}.jpg'.format(num), 'JPEG')
        num += 1


def combine():
    import numpy as np

    from PIL import Image
    # 这里是需要合并的图片路径
    paths = ["out{}.jpg".format(i) for i in range(1, 14)]
    img_array = ''
    img = ''
    for i, v in enumerate(paths):
        if i == 0:
            img = Image.open(v)  # 打开图片
            img_array = np.array(img)  # 转化为np array对象
        if i > 0:
            img_array2 = np.array(Image.open(v))
            img_array = np.concatenate((img_array, img_array2), axis=1)  # 横向拼接
            # img_array = np.concatenate((img_array, img_array2), axis=0)  # 纵向拼接
            img = Image.fromarray(img_array)

    # 保存图片
    img.save('图1.jpg')
