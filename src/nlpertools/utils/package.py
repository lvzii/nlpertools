#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
from importlib import import_module


def try_import(name, package):
    try:
        return import_module(name, package=package)
    except:
        print("import {} failed".format(name))
    finally:
        pass


# import aioredis
# import happybase
# import pandas as pd
# import pymysql
# from elasticsearch import Elasticsearch, helpers
# from kafka import KafkaProducer, KafkaConsumer
# from pymongo import MongoClient
# from elasticsearch import helpers
# from kafka import KafkaConsumer
aioredis = try_import("aioredis", None)
happybase = try_import("happybase", None)
pd = try_import("pandas", None)
pymysql = try_import("pymysql", None)
Elasticsearch = try_import("elasticsearch", "Elasticsearch")
KafkaProducer = try_import("kafka", "KafkaProducer")
MongoClient = try_import("pymongo", "MongoClient")
helpers = try_import("elasticsearch", "helpers")
KafkaConsumer = try_import("kafka", "KafkaConsumer")
# import numpy as np
# import seaborn as sns
# import torch
# import torch.nn as nn
# import xgboost as xgb
# from matplotlib import pyplot as plt
# from nltk.stem import WordNetLemmatizer
# from sklearn import metrics
# from transformers import BertTokenizer, BertForMaskedLM
# from transformers import BertForMaskedLM
np = try_import("numpy", None)
sns = try_import("seaborn", None)
torch = try_import("torch", None)
nn = try_import("torch.nn", None)
xgb = try_import("xgboost", None)
plt = try_import("matplotlib", "pyplot")
WordNetLemmatizer = try_import("nltk.stem", "WordNetLemmatizer")
metrics = try_import("sklearn", "metrics")
BertTokenizer = try_import("transformers", "BertTokenizer")
BertForMaskedLM = try_import("transformers", "BertForMaskedLM")
# import requests
requests = try_import("requests", None)
# import numpy as np
# import psutil
# import pyquery as pq
# import requests
# import torch
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
# from sklearn.metrics import precision_recall_fscore_support
# from tqdm import tqdm
# from win32evtlogutil import langid
# from sklearn.feature_extraction.text import TfidfTransformer
psutil = try_import("psutil", None)
pq = try_import("pyquery", None)
CountVectorizer = try_import("sklearn.feature_extraction.text", "CountVectorizer")
precision_recall_fscore_support = try_import("sklearn.metrics", "precision_recall_fscore_support")
tqdm = try_import("tqdm", "tqdm")
langid = try_import("win32evtlogutil", "langid")
TfidfTransformer = try_import("sklearn.feature_extraction.text", "TfidfTransformer")
# import pandas as pd
# import yaml
yaml = try_import("yaml", None)
