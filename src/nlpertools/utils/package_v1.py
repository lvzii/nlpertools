#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import importlib
from importlib import import_module
from importlib.util import LazyLoader
from .lazy import lazy_module

EXCLUDE_LAZYIMPORT = {"torch", "torch.nn", "numpy"}


def try_import(name, package):
    try:
        if package:
            # print("import {} success".format(name))
            return lazy_module("{}.{}".format(package, name))
        else:
            if name in EXCLUDE_LAZYIMPORT:
                return import_module(name, package=package)
            return lazy_module(name)
        # return import_module(name, package=package)
    except:
        pass
        print("import {} failed".format(name))
    finally:
        pass


def lazy_import(importer_name, to_import):
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
langid = try_import("langid", None)
# win32evtlogutil?
TfidfTransformer = try_import("sklearn.feature_extraction.text", "TfidfTransformer")
yaml = try_import("yaml", None)
