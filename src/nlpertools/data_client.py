# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import datetime
import json
import logging

from . import DB_CONFIG_FILE
from .io.file import read_yaml
from .utils.package import *

# import aioredis
# import happybase
# import pandas as pd
# import pymysql
# from elasticsearch import Elasticsearch, helpers
# from kafka import KafkaProducer, KafkaConsumer
# from pymongo import MongoClient

logger = logging.getLogger(__name__)

global_db_config = read_yaml(DB_CONFIG_FILE)


class Neo4jOps(object):
    # neo4j 连接的超时秒数
    # py2neo 内部会重试 3 次...
    NEO4J_TIMEOUT = 0.3
    pass

class SqliteOps(object):
    import sqlite3
    database_path = r'xx.db'
    conn = sqlite3.connect(database_path)
    c = conn.cursor()
    sql = "select name from sqlite_master where type='table' order by name"
    c.execute(sql)
    print(c.fetchall())
    sql = "select * from typecho_contents"
    c.execute(sql)
    res = c.fetchall()
    print(res[3])

    conn.commit()
    conn.close()

class MysqlOps(object):
    import pandas as pd
    def __init__(self, config=global_db_config["mysql"]):
        self.db = pymysql.connect(host=config["host"],
                                  port=config["port"],
                                  user=config["user"],
                                  password=config["password"],
                                  database=config["database"])

    def query(self, sql):
        df = pd.read_sql(sql, self.db)
        return df


class EsOps(object):
    from elasticsearch import Elasticsearch, helpers
    def __init__(self, config=global_db_config["es"]):
        self.es = Elasticsearch(
            host=config["host"], timeout=config["timeout"])

    def search_roll(self, index, body):
        all_data = []
        data = self.es.search(index=index, body=body, scroll="5m")
        all_data.extend(data["hits"]["hits"])
        scroll_id = data["_scroll_id"]
        while data["hits"]["hits"]:
            print(scroll_id[:5])
            data = self.es.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = data["_scroll_id"]
            all_data.extend(data["hits"]["hits"])
        all_data = [i["_source"] for i in all_data]
        return all_data

    def search_roll_iter(self, index, body):
        data = self.es.search(index=index, body=body, scroll="5m")
        scroll_id = data["_scroll_id"]
        while data["hits"]["hits"]:
            yield data["hits"]["hits"]
            data = self.es.scroll(scroll_id=scroll_id, scroll="5m")
            scroll_id = data["_scroll_id"]

    def search(self, index, body):
        return self.es.search(index=index, body=body)

    def delete(self, index, body):
        self.es.delete_by_query(index=index, body=body)

    def save(self, data):
        # data里有index
        helpers.bulk(self.es, data)

    def delete_data_by_query(self, index, _project_id, _source_ids):
        _query = {
            "query": {
                "bool": {
                    "must": [
                        {"terms": {"source_id": _source_ids}},
                        {"term": {"project_id": _project_id}},
                    ]
                }
            }
        }
        _res = self.es.delete_by_query(index=index, body=_query)
        print(f"delete_data_by_query: {_res}")

    def batch_re_save(self, index, _data, _project_id, _source_ids):
        self.delete_data_by_query(_project_id, _source_ids)
        _action = [{"_index": index, "_source": i} for i in _data]
        _res = helpers.bulk(self.es, _action)
        print(f"批量保存数据： {_res}")


class MongoOps(object):
    from pymongo import MongoClient
    def __init__(self, config=global_db_config["mongo"]):
        mongo_client = MongoClient(config["uri"])
        db = mongo_client[config["db"]]
        self.collection = db[config["col"]]

    def fetch_all(self):
        """
        读取所有数据
        :return:
        """
        ans = []
        print('提取所有数据.')
        for record in self.collection.find({}):
            record['_id'] = str(record['_id'])
            ans.append(record)
        return ans

    def load_from_mongo(self, special_value):
        """
        读取mongodb该special_value下所有值为special_value的数据
        :param
        :return:
        """
        record = self.collection.find({"{}".format(special_value): special_value})
        record = list(record)
        if not record:
            return None
        else:
            record = sorted(record, key=lambda x: len(x.get("another_value", [])))[0]
            return record

    def delete_all(self):
        query = {}
        deleted = self.collection.delete_many(query)
        return deleted

    def delete_by_time(self, time):
        query = {"name": {"$regex": "^F"}}
        deleted = self.collection.delete_many(query)

    def fetch_by_time(self, year=2022, month=7, day=7, hour=7, minute=7, second=7):
        query = {"query_time": {"$gte": datetime.datetime(year, month, day, hour, minute, second)}}
        sort_sql = [("query_time", -1)]
        ans = []
        print('提取所有数据.')
        for record in self.collection.find(query).sort(sort_sql):
            record['_id'] = str(record['_id'])
            ans.append(record)
        return ans

    def save_to_mongo(self, special_value, each_item):
        """
        数据存入mongo
        :param special_value:
        :param each_item:
        :return:
        """
        query = self.collection.find({"{}".format(special_value): special_value})
        if list(query):
            self.collection.update_one({"{}".format(special_value): special_value},
                                       {"$push": {'each_item': each_item}})
        else:
            insert_item = {
                "special_value": special_value,
                "each_item": [each_item]
            }
            self.collection.insert_one(insert_item)
        print("update success")

    def insert_one(self, data):
        self.collection.insert_one(data)

    def update_to_mongo(self, condition_term, condition_value, new_value):
        """
        根据提供的字段和值，查询出对应的数据，更新数据存入mongo
        类似 updata
        :param condition_term: 条件字段term
        :param condition_value: 条件字段值
        :param new_value: 新的值。最好是dict，不是dict的话不知道行不行
        :return:
        """
        query = self.collection.find({condition_term: condition_value})
        if list(query):
            self.collection.update_one({condition_term: condition_value},
                                       {"$push": new_value})
        else:
            insert_item = {
                condition_term: condition_value,
                "processed_data": new_value
            }
            self.collection.insert_one(insert_item)
        print("update success")


class RedisOps(object):
    def __init__(self, config=global_db_config["redis"]):
        redis_max_connections = 1024
        REDIS_GET_TIMEOUT = 0.1
        self.redis = aioredis.from_url(config["uri"], max_connections=redis_max_connections)


class HBaseOps(object):
    import happybase
    """
    demo
    key = 'test'
    db = HBaseHelper(host=hbase_host)
    data = db.query_single_line(table='table', row_key=key)
    print(data)
    """

    def __init__(self, config=global_db_config["hbase"]):
        self.host = config["DEFAULT_HOST"]
        self.port = config["DEFAULT_PORT"]
        self.compat = config["DEFAULT_COMPAT"]
        self.table_prefix = None  # namespace
        self.transport = config["DEFAULT_TRANSPORT"]
        self.protocol = config["DEFAULT_PROTOCOL"]
        self.conn = self.connect()

    def connect(self):
        conn = happybase.Connection(host=self.host, port=self.port, timeout=None, autoconnect=True,
                                    table_prefix=self.table_prefix, compat=self.compat,
                                    transport=self.transport, protocol=self.protocol)
        return conn

    def create_hb_table(self, table_name, **families):
        self.conn.create_table(table_name, families)

    def single_put(self, table_name, row_key, column, data):
        hb = happybase.Table(table_name, self.conn)
        hb.put(row_key,
               data={'{column}:{k}'.format(column=column, k=k): str(v).encode("utf-8") for k, v in data.items()})

    def batch_put(self, table, row_key_name, column, datas, batch_size=1):
        hb = happybase.Table(table, self.conn)
        datas_new = [datas[i:i + batch_size] for i in range(0, len(datas), batch_size)]
        for x in datas_new:
            with hb.batch(batch_size=batch_size) as batch:
                for da in x:
                    da_nw = {'{column}:{k}'.format(column=column, k=k): v for k, v in da.items()}
                    row_key = da_nw.pop('{column}:{k}'.format(column=column, k=row_key_name))
                    batch.put(row_key, da_nw)
        return batch

    def single_put_self(self, table_name, row_keys, datas):
        hb = happybase.Table(table_name, self.conn)
        for row_key, (_, val) in zip(row_keys, datas.items()):
            hb.put(row_key, {'maybe_table_name:maybe_column_name': "%s" % val[0],
                             'maybe_table_name:maybe_column_name2': "%s" % val[1]})

    def scan_table(self, table, row_start=None, row_stop=None, include_timestamp=False, limit=None, timestamps=None,
                   filter=None):
        hb = happybase.Table(table, self.conn)
        scan = hb.scan(row_start=row_start, row_stop=row_stop, limit=limit, timestamp=timestamps, filter=filter)
        hb_dict = dict(scan)
        if hb_dict:
            return {str(k1).decode('utf-8'): {str(k2).decode('utf-8'): str(v2).decode('utf-8') for k2, v2 in v1.items()}
                    for k1, v1 in
                    hb_dict.items()}
        else:
            return {}

    def query_single_line(self, table, row_key):
        conn = self.connect()
        hb = happybase.Table(table, conn)
        hb_dict = hb.row(row_key)
        if hb_dict:
            return {k.decode('utf-8'): v.decode('utf-8') for k, v in hb_dict.items()}
        else:
            return {}

    def query_multi_lines(self, table, row_keys):
        hb = happybase.Table(table, self.conn)
        hb_dict = dict(hb.rows(row_keys))
        if hb_dict:
            return {k1.decode('utf-8'): {k2.decode('utf-8'): v2.decode('utf-8') for k2, v2 in v1.items()} for k1, v1 in
                    hb_dict.items()}
        else:
            return {}

    def single_delete(self, table, row_key):
        hb = happybase.Table(table, self.conn)
        hb.delete(row_key)

    def test_scan(self, table):
        hb = happybase.Table(table, self.conn)
        filter = "SingleColumnValueFilter ('maybe_column_name', 'lang', =, 'regexstring:[regex_string]')"
        scan = hb.scan(limit=1000, filter=filter)

        hb_dict = dict(scan)
        if hb_dict:
            return {str(k1).decode('utf-8'): {str(k2).decode('utf-8'): str(v2).decode('utf-8') for k2, v2 in v1.items()}
                    for k1, v1 in
                    hb_dict.items()}
        else:
            return {}

    def close(self):
        self.conn.close()


class KafkaConfig():
    pass


class KafkaOps(object):
    def __init__(self, config=global_db_config["kafka"]):
        self.bootstrap_server = config["bootstrap_server"]
        self.topic = config["topic"]
        # 超时时间设置默认30s， 修改为60s
        self.producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                      bootstrap_servers=self.bootstrap_server,
                                      acks='all',
                                      request_timeout_ms=60000)

    def send_data_to_kafka(self, data):
        try:
            self.producer.send(self.topic, data)
            logger.info(f"data send successful! ---- {data}")
        except Exception as e:
            logger.exception(f'kafka occur error ---- {e}')

    def consumer_msg(self):
        consumer = KafkaConsumer(self.topic, group_id='test-group_id', bootstrap_servers=self.bootstrap_server)
        for msg in consumer:
            recv = "%s:%d:%d: key=%s value=%s" % (msg.topic, msg.partition, msg.offset, msg.key, msg.value)
            print(recv)




class MilvusOps(object):
    def __init__(self, config=global_db_config.milvus):
        from pymilvus import connections, Collection

        connections.connect("default", host=config.host, port=config.port)
        self.collection = Collection(config.collection)
        self.collection.load()

    def get_similarity(self, embedding):
        search_params = {
            "metric_type": "L2",
            "params": {"nprobe": 1},
        }
        # # %%
        logger.debug(embedding)
        result = self.collection.search(
            [list(embedding)],
            "vec",
            search_params,
            limit=3,
            output_fields=["pk", "entity_name", "standard_entity_name"],
        )
        hits = result[0]
        entities = []
        for hit in hits:
            entities.append(
                {
                    "name": hit.entity.get("entity_name"),
                    "standard_name": hit.entity.get("standard_entity_name"),
                }
            )
        return entities

    # def insert(self, collection, entities):
    #     collection.insert(entities)
