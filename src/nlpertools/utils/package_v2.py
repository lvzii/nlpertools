# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import importlib
from importlib import import_module
import os


def try_import(name, package):
    try:
        return import_module(name, package=package)
    except:
        pass
        # print("import {} failed".format(name))
    finally:
        pass


aioredis = None
happybase = None
pd = None
pymysql = None
Elasticsearch = None
KafkaProducer = None
MongoClient = None
helpers = None
KafkaConsumer = None
np = None
sns = None
torch = None
nn = None
xgb = None
plt = None
WordNetLemmatizer = None
metrics = None
BertTokenizer = None
BertForMaskedLM = None
requests = None
psutil = None
pq = None
CountVectorizer = None
precision_recall_fscore_support = None
tqdm = None
langid = None
win32evtlogutil = None
TfidfTransformer = None
yaml = None

import_dict = {
    "aioredis": ("aioredis", None),
    "happybase": ("happybase", None),
    "pd": ("pandas", None),
    "pymysql": ("pymysql", None),
    "Elasticsearch": ("elasticsearch", "Elasticsearch"),
    "KafkaProducer": ("kafka", "KafkaProducer"),
    "MongoClient": ("pymongo", "MongoClient"),
    "helpers": ("elasticsearch", "helpers"),
    "KafkaConsumer": ("kafka", "KafkaConsumer"),
    "np": ("numpy", None),
    "sns": ("seaborn", None),
    "torch": ("torch", None),
    "nn": ("torch.nn", None),
    "xgb": ("xgboost", None),
    "plt": ("matplotlib", "pyplot"),
    "WordNetLemmatizer": ("nltk.stem", "WordNetLemmatizer"),
    "metrics": ("sklearn", "metrics"),
    "BertTokenizer": ("transformers", "BertTokenizer"),
    "BertForMaskedLM": ("transformers", "BertForMaskedLM"),
    "requests": ("requests", None),
    "psutil": ("psutil", None),
    "pq": ("pyquery", None),
    "CountVectorizer": ("sklearn.feature_extraction.text", "CountVectorizer"),
    "precision_recall_fscore_support": ("sklearn.metrics", "precision_recall_fscore_support"),
    "tqdm": ("tqdm", "tqdm"),
    "langid": ("langid", None),
    "win32evtlogutil": ("win32evtlogutil", None),
    "TfidfTransformer": ("sklearn.feature_extraction.text", "TfidfTransformer"),
    "yaml": ("yaml", None)
}
if "nlpertools_helper" in os.environ.keys():
    # TODO 该方法未经过测试
    import_list = os.environ["nlpertools_helper"]

    for k in import_list:
        name, package = import_dict[k]
        globals()[k] = try_import(name, package)
else:
    aioredis = try_import("aioredis", None)
    happybase = try_import("happybase", None)
    pd = try_import("pandas", None)
    pymysql = try_import("pymysql", None)
    Elasticsearch = try_import("elasticsearch", "Elasticsearch")
    KafkaProducer = try_import("kafka", "KafkaProducer")
    MongoClient = try_import("pymongo", "MongoClient")
    helpers = try_import("elasticsearch", "helpers")
    KafkaConsumer = try_import("kafka", "KafkaConsumer")
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
    requests = try_import("requests", None)
    psutil = try_import("psutil", None)
    pq = try_import("pyquery", None)
    CountVectorizer = try_import("sklearn.feature_extraction.text", "CountVectorizer")
    precision_recall_fscore_support = try_import("sklearn.metrics", "precision_recall_fscore_support")
    tqdm = try_import("tqdm", "tqdm")
    # TODO 自动导出langid和win32evtlogutil输出有bug
    langid = try_import("langid", None)
    win32evtlogutil = try_import("win32evtlogutil", None)
    TfidfTransformer = try_import("sklearn.feature_extraction.text", "TfidfTransformer")
    yaml = try_import("yaml", None)
