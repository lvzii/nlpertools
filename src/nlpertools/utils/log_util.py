#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
"""
# 该项目暂时没有日志输出
import codecs
import logging.config

import nlpertools
import yaml

nlpertools.j_mkdir("logs")

with codecs.open('log_config.yml', 'r', 'utf-8') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

# logging.basicConfig(level=logging.INFO)
logging.config.dictConfig(config)
logger = logging.getLogger()
"""