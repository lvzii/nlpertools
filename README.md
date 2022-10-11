# Current version

1.0.5

# 备注

最新版本在nlpertools里，不在src/nlpertools里

# Introduction

This is a package of functions commonly used by NLP/data workers, which can simplify some reading and writing operations
and make the code more readable. It mainly includes two parts: basic reading and writing tools and data processing
functions commonly used in machine learning/deep learning.

```bash
io # 基本的读写工具。包括了文件读写、文件夹读写、数据读写、词频统计等功能。
  dir # 文件夹读写
  file  # 文件读写
  other # 其他
ml # 机器学习/深度学习工作中常用的数据处理函数。包括划分十折交叉数据、常见json格式数据读取等功能。
openApi # 网络开源接口
plugin  # 常用插件
reminder  # 提示
other # 其他一些未分组功能
```

## Document

https://nlpertools.readthedocs.io/en/latest/

## Development Guide

- import 全部放在一个文件里面，所有都从里面找，通过utils可以输出所用到的所有import

- 然后运行脚本，生成打包到pypi里的nlpertools

- 因为里面有很多的引用，需要依赖，不可能全部都安装

- 类似paddle、ltp的import需要判断是否使用才import，因为import的时间太长
  例子：

- [git commit guide](https://blog.csdn.net/fd2025/article/details/124543690)

```python
class STEM(object):
    from ltp import LTP

    def __init__(self, IPT_MODEL_PATH):
        self.ltp = LTP(IPT_MODEL_PATH)
```

## Emample

```python
import nlpertools

a = nlpertools.readtxt_list_all_strip('res.txt')
# 或
b = nlpertools.io.file.readtxt_list_all_strip('res.txt')
```


pypi用
- setup.cfg
- pyproject.toml

文档用
- mkdocs.yml
- .readthedocs.yml
