<div align="center">
  <h4 align="center">
      <p>
          <a href="https://github.com/lvzii/nlpertools/blob/main/README.md">中文</a> |
          <b>English</b>
      </p>
  </h4>
</div>


# Introduction

This is a package of functions commonly used by NLP/data workers, which can simplify some reading and writing operations
and make the code more readable. It mainly includes two parts: basic reading and writing tools and data processing
functions commonly used in machine learning/deep learning.

What problem does it solve:

- Many functions are hard to remember, and every time you need to search or ask a large model, e.g., pandas sorting.
- When practicing coding problems, tree structure problems are hard to debug.

```
nlpertools
 ├── mkdocs.yml # used in doc
 ├── .readthedocs.yml # used in doc
 ├── pyproject.toml # used in pypi
 └── setup.cfg # used in pypi
```

# Most commonly used/favorite features (usage examples)

```python
# Read txt, json files
import nlpertools

txt_data = nlpertools.readtxt_list_all_strip('res.txt')
json_data = nlpertools.load_from_json('res.json')
```

```bash
## git, very useful when github connection is unstable
ncli git pull

# Generate real-time key for pypi two-factor authentication (need to provide key)
ncli --get_2fa --get_2fa_key your_key
```

# 


# Install

Install the latest release version

```bash
pip install nlpertools
```

📢[ Recommend ] Install the latest git version
```
pip install git+https://github.com/lvzii/nlpertools
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
    According to the assessment by `pyinstrument`, packages that exceed 1 second will adopt this method.
  - 2s+ happybase、seaborn、torch、jieba
  - 1s+ /
  - 0.5s+ pandas elasticsearch transformers xgboost nltk mongo
  
- [git commit guide](https://blog.csdn.net/fd2025/article/details/124543690)

- [upload to pypi](https://juejin.cn/post/7369413136224878644)

- [readthedoc:check doc server status](https://readthedocs.org/projects/nlpertools/builds)

## Development philosophy

Write a function once, never write it twice!


## Contribution

https://github.com/bigscience-workshop/data-preparation

