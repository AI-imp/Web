from pymongo import MongoClient
from .data import setting
client = MongoClient(host=setting.MONGODB_HOST, port=setting.MONGODB_PORT)
mydb = client.Policy_search
collection = mydb.Modified_data
rows = collection.find()
s=0
for row in rows:
    a=row['time']
    s=s+1
    print(s)
