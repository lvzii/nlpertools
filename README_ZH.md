# 当前版本
1.0.5

# 说明
这是一些NLP/数据工作人员常用的函数组成的包，可以简化一些读写操作，使代码更加可读。主要包括两个部分：基本的读写工具和机器学习/深度学习工作中常用的数据处理函数。
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
## 开发指南

- 需要再nlpertools_dev里开发

- 然后运行脚本，生成打包到pypi里的nlpertools

- 因为里面有很多的引用，需要依赖，不可能全部都安装

- [git 提交信息参考](https://blog.csdn.net/fd2025/article/details/124543690)

- 类似paddle、ltp的import需要判断是否使用才import，因为import的时间太长

例子：
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

## 提交参考：

https://blog.csdn.net/fd2025/article/details/124543690
