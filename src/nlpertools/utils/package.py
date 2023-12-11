#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import importlib
from importlib import import_module


def try_import(name, package):
    try:
        return import_module(name, package=package)
    except:
        pass
        # print("import {} failed".format(name))
    finally:
        pass


def lazy_import(importer_name, to_import):
    """
    Example from net
    author: unknown
    this function is not used
    """
    """Return the importing module and a callable for lazy importing.

    The module named by importer_name represents the module performing the
    import to help facilitate resolving relative imports.

    to_import is an iterable of the modules to be potentially imported (absolute
    or relative). The `as` form of importing is also supported,
    e.g. `pkg.mod as spam`.

    This function returns a tuple of two items. The first is the importer
    module for easy reference within itself. The second item is a callable to be
    set to `__getattr__`.
    """
    module = importlib.import_module(importer_name)
    import_mapping = {}
    for name in to_import:
        importing, _, binding = name.partition(' as ')
        if not binding:
            _, _, binding = importing.rpartition('.')
        import_mapping[binding] = importing

    def __getattr__(name):
        if name not in import_mapping:
            message = f'module {importer_name!r} has no attribute {name!r}'
            raise AttributeError(message)
        importing = import_mapping[name]
        # imortlib.import_module() implicitly sets submodules on this module as
        # appropriate for direct imports.
        imported = importlib.import_module(importing,
                                           module.__spec__.parent)
        setattr(module, name, imported)
        return imported

    return module, __getattr__


# jieba = try_import("jieba", None)
# sns = try_import("seaborn", None)
# torch = try_import("torch", None)
# nn = try_import("torch.nn", None)
# BertTokenizer = try_import("transformers", "BertTokenizer")
# BertForMaskedLM = try_import("transformers", "BertForMaskedLM")
# Elasticsearch = try_import("elasticsearch", "Elasticsearch")
# pd = try_import("pandas", None)
# xgb = try_import("xgboost", None)

aioredis = try_import("aioredis", None)
pymysql = try_import("pymysql", None)
zhconv = try_import("zhconv", None)
KafkaProducer = try_import("kafka", "KafkaProducer")
KafkaConsumer = try_import("kafka", "KafkaConsumer")
np = try_import("numpy", None)
plt = try_import("matplotlib", "pyplot")
WordNetLemmatizer = try_import("nltk.stem", "WordNetLemmatizer")
metrics = try_import("sklearn", "metrics")
requests = try_import("requests", None)
pq = try_import("pyquery", None)
CountVectorizer = try_import("sklearn.feature_extraction.text", "CountVectorizer")
precision_recall_fscore_support = try_import("sklearn.metrics", "precision_recall_fscore_support")
tqdm = try_import("tqdm", "tqdm")
# TODO 自动导出langid和win32evtlogutil输出有bug
langid = try_import("langid", None)
win32evtlogutil = try_import("win32evtlogutil", None)
TfidfTransformer = try_import("sklearn.feature_extraction.text", "TfidfTransformer")
yaml = try_import("yaml", None)
omegaconf = try_import("omegaconf", None)