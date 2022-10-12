# Current version

1.0.5

# Introduction

This is a package of functions commonly used by NLP/data workers, which can simplify some reading and writing operations
and make the code more readable. It mainly includes two parts: basic reading and writing tools and data processing
functions commonly used in machine learning/deep learning.

```
nlpertools
 ├── mkdocs.yml # used in doc
 ├── .readthedocs.yml # used in doc
 ├── pyproject.toml # used in pypi
 └── setup.cfg # used in pypi
```

## Document

https://nlpertools.readthedocs.io/en/latest/

## Development Guide

- `import` op is placed in utils/package.py

- `paddle`, `ltp`, etc. import time is too long and requires other import methods:
  ```python
  class STEM(object):
      from ltp import LTP
  
      def __init__(self, IPT_MODEL_PATH):
          self.ltp = LTP(IPT_MODEL_PATH)
  ```
- [git commit guide](https://blog.csdn.net/fd2025/article/details/124543690)

- [readthedoc:check doc server status](https://readthedocs.org/projects/nlpertools/builds)

## Emample

```python
import nlpertools

a = nlpertools.readtxt_list_all_strip('res.txt')
# 或
b = nlpertools.io.file.readtxt_list_all_strip('res.txt')
```

