#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import math

import faiss
import gensim
import numpy as np
import pandas as pd


def build_index_use(vectors):
    d = len(vectors[0])
    nlist = 100
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
    index.train(vectors)
    index.add(vectors)
    return index


def build_index(vectors, distances="L2", nprobe=10):
    """ 建立 faiss 索引.

    Args:
        vectors(numpy.array): 向量矩阵，shape=(向量数, 向量维度)
        distance(str): 度量距离，支持 L2、COS 和 INNER_PRODUCT.
        nprobe(int): 向量搜索时需要搜索的聚类数.

    Return: 返回 faiss 索引对象.

    """
    metric_type = None
    if distances == "L2":
        metric_type = faiss.METRIC_L2
    elif distances in ("COS", "INNER_PRODUCT"):
        metric_type = faiss.METRIC_INNER_PRODUCT
    else:
        raise NotImplementedError

    index_pipes = []

    if distances == "COS":
        index_pipes.append("L2norm")

    K = 4 * math.sqrt(vectors.shape[0])
    use_ivf = False
    if vectors.shape[0] >= 30 * K:
        index_pipes.append(f"IVF{K}")
        use_ivf = True

    index_pipes.append("Flat")

    index = faiss.index_factory(vectors.shape[1], ",".join(index_pipes),
                                metric_type)

    vectors = vectors.astype(np.float32)
    if not index.is_trained:
        index.train(vectors)

    index.add(vectors)

    # IVF 使用 reconstruct 时必须执行此函数
    if use_ivf:
        ivf_index = faiss.extract_index_ivf(index)
        ivf_index.make_direct_map()
        ivf_index.nprobe = nprobe

    return index


def read_index_from_file(filename):
    """ 从向量文件中读取 faiss 向量对象. """
    return faiss.read_index(filename)


def write_index_to_file(index, filename):
    """ 将 faiss 向量对象写入文件. """
    faiss.write_index(index, filename)


word2vec_path = "glove_vector_path"
wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, no_header=True)
vectors = wv_from_text.vectors

name_example = ["{}.jpg".format((i % 9) + 1) for i in range(len(vectors))]
df = pd.DataFrame({
    "name": name_example,
    # "vector": str(vectors[0]),
    # "text": list(wv_from_text.key_to_index.keys()),
})
test_index = build_index_use(vectors)
write_index_to_file(test_index, "test.index")

df.to_csv("test.csv", index=False)

import gensim
import hnswlib

word2vec_path = "glove_vector_path"
wv_from_text = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=False, no_header=True)
vectors = wv_from_text.vectors

labels = [idx for idx, i in enumerate(vectors)]
index = hnswlib.Index(space="l2", dim=len(vectors[0]))
index.init_index(max_elements=len(vectors), ef_construction=200, M=16)
index.add_items(vectors, labels)
index.save_index("hnswlib.index")
