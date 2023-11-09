<div align="center">
  <h4 align="center">
      <p>
          <b>中文</b> |
          <a href="https://github.com/lvzii/nlpertools/blob/main/README_EN.md">English</a>
      </p>
  </h4>
</div>


# 当前版本

1.0.5

# 说明

这是一些NLP/数据工作人员常用的函数组成的包，可以简化一些读写操作，使代码更加可读。主要包括两个部分：基本的读写工具和机器学习/深度学习工作中常用的数据处理函数。

它解决了什么问题：

- 很多函数是记不住的，每次写每次都要搜，例如pandas排序
- 刷题的时候，树结构的题目很难调试


```
nlpertools
 ├── mkdocs.yml # used in doc
 ├── .readthedocs.yml # used in doc
 ├── pyproject.toml # used in pypi
 └── setup.cfg # used in pypi

```

## 开发指南

- import都放在了utils/package.py里，通过脚本可自动生成

- 类似paddle、ltp的import需要判断是否使用才import，因为import的时间太长，例如:
  ```python
  class STEM(object):
      from ltp import LTP
  
      def __init__(self, IPT_MODEL_PATH):
          self.ltp = LTP(IPT_MODEL_PATH)
  ```
  通过pyinstrument判断，超过1s的包即采用这种方式
  - 2s+ happybase、seaborn、torch、jieba
  - 1s+
  - 0.5s+ pandas elasticsearch transformers xgboost nltk mongo


- [git commit guide](https://blog.csdn.net/fd2025/article/details/124543690)

- [readthedoc 检查文档构建状况](https://readthedocs.org/projects/nlpertools/builds)

- 发布版本需要加tag

## 开发哲学

针对读取文件的方法，是将一些参数直接写在函数里，以实现快速使用。

原则是：写过一遍的函数，绝不写第二遍！

一些可能需要配置才能用的函数，写上示例

## 

```python
import nlpertools

a = nlpertools.readtxt_list_all_strip('res.txt')
# 或
b = nlpertools.io.file.readtxt_list_all_strip('res.txt')
```
