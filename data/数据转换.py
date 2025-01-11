from pymongo import MongoClient
from data import setting
client = MongoClient(host=setting.MONGODB_HOST, port=setting.MONGODB_PORT)
mydb = client.Policy_search
collection = mydb.Modified_data
rows = collection.find()
s=0
c=[]
for row in rows:
    a=row['PUB_TIME']
    b=a.split('/',3)
    s=s+1
    c=int(b[0])*10000+int(b[1])*100+int(b[2])
    myquery = {"POLICY_ID": row['POLICY_ID']}
    newvalues = {"$set": {"time": c}}

    collection.update_one(myquery, newvalues)

print(s)
