#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import nlpertools


# nlpertools.writetxt_w_list(['1'], "test.txt")
# def fun(a):
#     print(1)
# nlpertools.stress_test(fun, [1,2])

# # 遍历所有py文件，将所有外部依赖的 import 转换为try import
# nlpertools.convert_import_to_try_import()
# # 为便于开发，转化try import为正常import
# nlpertools.convert_try_import_to_import()
# # 提交
# nlpertools.git_push()
nlpertools.convert_import_to_import("./nlpertools")