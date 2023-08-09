package说明

# v0
即没有后缀的package.py
全部try，缺点是耗时长

# v1
是lazy import
但是针对torch numpy等包失败(推测是他、它们自己做了懒加载处理)

# v2
通过环境变量加载需要import的，即nlpertools_help

# v3
nlpertools里如果import了paddle等东西会巨慢,查到原因了是ltp,解决方法是用rasa的required_packages字段控制

# v4
上面所有的方案缺点就是没法静态检查， 无法追踪代码
可以通过对依赖分组，某些分组需要装包，某些分组不需要装，然后可以try某一个分组
该方案需要大改

