import pandas as pd
from pymongo import MongoClient

from data import setting

data = pd.read_csv('policyinfo(已修改).tsv', encoding='gb18030', sep='\t', low_memory=False)
data.fillna(" ",inplace=True)
'''
print(len(data))
print(data.columns)
print(data.tail)
print(data.iloc[6]['POLICY_TITLE'])
print(data)
print(data.iloc[9])
'''
client = MongoClient(host=setting.MONGODB_HOST, port=setting.MONGODB_PORT)
mydb = client.Policy_search
collection = mydb.Modified_data
for index, row in data.T.to_dict().items():
  collection.insert_one(row)