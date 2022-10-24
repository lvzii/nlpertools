package说明

# v0
即没有后缀的package.py
全部try，缺点是耗时长

# v1
是lazy import
但是针对torch numpy等包失败(推测是他、它们自己做了懒加载处理)

# v2
通过环境变量加载需要import的