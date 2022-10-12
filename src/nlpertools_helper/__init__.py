#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import os

import_list = []


def save_import_list_to_environment(import_list):
    os.environ["nlpertools_helper"] = import_list
