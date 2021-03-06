# !/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Author  : youshu.Ji
import json
import logging

import aioredis
import happybase
import pandas as pd
import pymysql
from elasticsearch import Elasticsearch
from kafka import KafkaProducer, KafkaConsumer
from pymongo import MongoClient

from . import DB_CONFIG_FILE
from .io.file import read_yaml

logger = logging.getLogger(__name__)

global_db_config = read_yaml(DB_CONFIG_FILE)


class Neo4jOps(object):
    # neo4j 连接的超时秒数
    # py2neo 内部会重试 3 次...
    NEO4J_TIMEOUT = 0.3
    pass


class MysqlOps(object):
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

    def search(self, index, body):
        return self.es.search(index=index, body=body)


class MongoOps(object):
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

    def delete_by_time(self, time):
        query = {"name": {"$regex": "^F"}}
        deleted = self.collection.delete_many(query)

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


class RedisOps(object):
    def __init__(self, config=global_db_config["redis"]):
        REDIS_MAX_CONNECTIONS = 1024
        REDIS_GET_TIMEOUT = 0.1
        self.redis = aioredis.from_url(config["uri"], max_connections=REDIS_MAX_CONNECTIONS)


class HBaseOps(object):
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


class KafkaOps():
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



