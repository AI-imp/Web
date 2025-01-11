from whoosh.fields import Schema, ID, TEXT
from whoosh.index import create_in, open_dir
# from whoosh.query import *
# from whoosh.qparser import *
from jieba.analyse import ChineseAnalyzer
import pymongo
from pymongo.collection import Collection
from whoosh.qparser import QueryParser

from data import setting


class IndexBuilder:
    def __init__(self):
        self.mongoClient = pymongo.MongoClient(host=setting.MONGODB_HOST, port=setting.MONGODB_PORT)
        # self.db = self.mongoClient[settings.MONGODB_DBNAME][settings.MONGODB_SHEETNAME]
        self.db = pymongo.database.Database(self.mongoClient, setting.MONGODB_DBNAME)
        self.pagesCollection = Collection(self.db, setting.MONGODB_SHEETNAME)

    def build_index(self):
        analyzer = ChineseAnalyzer()

        # 创建索引模板
        schema = Schema(
            id=ID(stored=True),
            POLICY_ID=ID(stored=True),
            POLICY_TITLE=TEXT(stored=True, analyzer=analyzer),
            POLICY_GRADE=TEXT(stored=True, analyzer=analyzer),
            PUB_AGENCY_ID=ID(stored=True),
            PUB_AGENCY=TEXT(stored=True, analyzer=analyzer),
            PUB_AGENCY_FULLNAME=TEXT(stored=True, analyzer=analyzer),

            PUB_TIME=TEXT(stored=True),
            POLICY_TYPE=TEXT(stored=True, analyzer=analyzer),
            POLICY_BODY=TEXT(stored=True, analyzer=analyzer),
            PROVINCE=TEXT(stored=True, analyzer=analyzer),
            CITY=TEXT(stored=True, analyzer=analyzer),
            POLICY_SOURCE=TEXT(stored=True, analyzer=analyzer),
            UPDATE_DATE=TEXT(stored=True),
            priority=TEXT(stored=True, analyzer=analyzer,sortable=True),
            time=TEXT(stored=True, analyzer=analyzer),
        )
        import os.path
        if not os.path.exists('index_'):
            os.mkdir('index_')
            ix = create_in('index_', schema)
            print('未发现索引文件,已构建.')
        else:
            ix = open_dir('index_')
            print('发现索引文件并载入....')

        # 索引构建
        writer = ix.writer()
        rows = self.pagesCollection.find()
        indexed_amount = 0
        for row in rows:
            indexed_amount += 1
            writer.add_document(
                id=str(row['_id']),
                POLICY_ID=row['POLICY_ID'],
                POLICY_TITLE=row['POLICY_TITLE'],
                POLICY_GRADE=row['POLICY_GRADE'],
                PUB_AGENCY_ID=row['PUB_AGENCY_ID'],
                PUB_AGENCY=row['PUB_AGENCY'],
                PUB_AGENCY_FULLNAME=row['PUB_AGENCY_FULLNAME'],

                PUB_TIME=row['PUB_TIME'],
                POLICY_TYPE=row['POLICY_TYPE'],
                POLICY_BODY=row['POLICY_BODY'],
                PROVINCE=row['PROVINCE'],
                CITY=row['CITY'],
                POLICY_SOURCE=row['POLICY_SOURCE'],
                UPDATE_DATE=row['UPDATE_DATE'],
                priority=str(row['priority']),
                time=str(row['time']),
            )
        writer.commit()
        print(indexed_amount)

# --------此段代码用以在数据库中缺少indexed字段时补充插入indexed字段并初始化为false--------
# host = settings.MONGODB_HOST
# port = settings.MONGODB_PORT
# dbname = settings.MONGODB_DBNAME
# sheetname = settings.MONGODB_SHEETNAME
# client = pymongo.MongoClient(host=host, port=port)
# mydb = client[dbname]
# post = mydb[sheetname]
# post.update({}, {'$set':{'indexed':'False'}}, upsert=False, multi=True)   # 增加indexed项并初始化为False
# post.update({'indexed': 'True'}, {'$set':{'indexed':'False'}})
# --------------------------------------------------------------------------------------


if __name__ == '__main__':
    id=IndexBuilder()
    id.build_index()
    ix = open_dir("index_")
    with ix.searcher() as searcher:
        query = QueryParser("POLICY_ID", ix.schema).parse("100223252")
        results = searcher.search(query)
        print(results[0])