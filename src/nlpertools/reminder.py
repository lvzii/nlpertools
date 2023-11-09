#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
from .utils.package import *


def remind_assert():
    assert "train_extension" in ["csv", "json"], "`train_file` should be a csv or a json file."


def remind_dir():
    reminder = "os.path.dirname(os.path.abspath(__file__))"


def remind_me():
    reminder = "-> 数据获取 -> 数据清洗  -> dataclean -> 预标注  -> get_TexSmart -> 校对 -> 添加数据训练 -> 评价 -> 纠正标注数据"


class PandasLookup:
    @staticmethod
    def merge_df(a, b):
        # example
        a = pd.DataFrame({
            "id": [1, 2]
        })
        b = pd.DataFrame({
            "id": [2, 3, 1],
            "content": ['b', 'c', 'a'],
        })
        merged = pd.DataFrame({
            "id": [1, 2],
            "content": ["a", 'b']
        })
        merged = a.merge(b, left_on='id', right_on='id', how='left')
        return merged


class OtherLookup:
    def prometheus_demo(self):
        return

    def load_two_dataset(self):
        from datasets import interleave_datasets
        from itertools import islice
        en_dataset = load_dataset('oscar', "unshuffled_deduplicated_en", split='train', streaming=True)
        fr_dataset = load_dataset('oscar', "unshuffled_deduplicated_fr", split='train', streaming=True)
        multilingual_dataset = interleave_datasets([en_dataset, fr_dataset])
        print(list(islice(multilingual_dataset, 2)))
        multilingual_dataset_with_oversampling = interleave_datasets([en_dataset, fr_dataset], probabilities=[0.8, 0.2],
                                                                     seed=42)
        print(list(islice(multilingual_dataset_with_oversampling, 2)))

    """
    import prometheus_client
    from flask import Response, Flask, request
    from flask_restful import Api, Resource
    from prometheus_client.core import Counter
    
    
    def create_app():
        app = Flask(__name__)
        return app
    
    
    app = create_app()
    api = Api(app)
    requests_total = Counter("request_count", "Total request count of the host")
    
    
    class getMetrics(Resource):
        def post(self):
            return Response(prometheus_client.generate_latest(),
                            mimetype="text/plain")
    
    
    class getInfo(Resource):
        def post(self):
            requests_total.inc()
            return {}
    
    
    api.add_resource(getInfo, '/test')
    api.add_resource(getMetrics, "/metrics")
    """

    def flask_download_demo(self):
        return

    """
    from flask import send_file, send_from_directory
    import os
    
    @app.route("/download/<filename>", methods=['GET'])
    def download_file(filename):
        # 需要知道2个参数, 第1个参数是本地目录的path, 第2个参数是文件名(带扩展名)
        directory = os.getcwd()  # 假设在当前目录
        return send_from_directory(directory, filename, as_attachment=True)
    """
