#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
# 定义装饰器
from functools import wraps


def fn_timer(function):
    @wraps(function)
    def function_timer(*args, **kwargs):
        t0 = time.time()
        result = function(*args, **kwargs)
        t1 = time.time()
        print('[finished {func_name} in {time:.2f}s]'.format(func_name=function.__name__, time=t1 - t0))
        return result

    return function_timer


def fn_timeout_checker(wait_time, callback):
    """
    超时判断的装饰器
    两个包，使用gevent出现bug
    """
    # from gevent import Timeout
    # from gevent.monkey import patch_all

    # patch_all() # thread=False加了这个参数，配合flask app的threaded=True,会报错，目前还没有理解阻塞，线程之间的关系。不加即thread=True时没问题

    from eventlet import Timeout
    from eventlet import monkey_patch

    monkey_patch(time=True)

    def wrapper(func):
        def inner(*args, **kwargs):
            finish_flag = False
            with Timeout(wait_time, False):
                res = func(*args, **kwargs)
                finish_flag = True
            if not finish_flag:
                res = callback()
            return res

        return inner

    return wrapper


def fn_try(parameter):
    """
    该函数把try...catch...封装成装饰器，
    接收一个字典参数，并把其中的msg字段改为具体报错信息
    :param parameter: {"msg": "", etc.}
    :return: parameter: {"msg": 内容填充为具体的报错信息, etc.}
    """

    def wrapper(function):
        def inner(*args, **kwargs):
            try:
                result = function(*args, **kwargs)
                return result
            except Exception as e:
                msg = "报错！"
                print('[func_name: {func_name} {msg}]'.format(func_name=function.__name__, msg=msg))
                parameter["msg"] = parameter["msg"].format(str(e))
                return parameter
            finally:
                pass

        return inner

    return wrapper


def example(function):
    @wraps(function)
    def function_example(*args, **kwargs):
        print("此方法仅仅用于提示该方法怎么写")
        result = function(*args, **kwargs)
        return result

    return function_example


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


