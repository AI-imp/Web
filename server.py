from flask import Flask, jsonify, request,render_template,redirect,url_for
from flask_cors import CORS
from pymongo import MongoClient
from django.shortcuts import render
import sys

from werkzeug.datastructures import ImmutableMultiDict
from django.shortcuts import render
import sys

from pymongo import MongoClient
res = None
sys.path.append('..')
from data.query import Query
app = Flask(__name__)
CORS(app)
q = Query('../data/index_')
@app.route('/')
def main():
    msg='登录'
    return render_template('main.html',msg=msg)

@app.route('/guest/<guest>')
def hello_guest(guest):
    msg="尊敬的"+str(guest)+"，您好"
    return render_template("main.html", msg=msg)
@app.route('/sign_in', methods=['POST','GET'])
def addData():
    data = request.form
    msg=''
    if data!=ImmutableMultiDict([]):
        print('--->upload form data: ', data)
        username = data.get('username', 'no data')
        password = data.get('password', '0')
        Email = data.get('Email', 'no data')
        telphone=data.get('telphone','0')
        sex = data.get('sex', 'no data')
        city=data.get('city','no data')
        prov=data.get('prov','no data')
        interest= data.getlist('interest')
        if username!='' and password!='' and Email!='' and telphone!=''and sex!='':
            client = MongoClient('mongodb://localhost:27017/')
            db = client.home
            collection = db.mine
            a={'username':username,'password':password,'telphone':telphone,'Email':Email,'sex':sex,'prov':prov,"city":city,"interest":interest}
            count = collection.count_documents({"telphone": telphone})
            print(count)
            if count==0:
                result = collection.insert_one(a)
                msg="用户注册成功"
                return redirect(url_for('checkData'))
            else:
                msg="账号已存在"
            return render_template("sign_in.html", msg=msg)
    return render_template("sign_in.html")

@app.route('/log_in', methods=['POST','GET'])
def checkData():
    data = request.form
    if data != ImmutableMultiDict([]):
        print('data: ', data)
        username = data.get('username', 'no data')
        password = data.get('password', '0')
        ip_pro = data.get('ip_pro', 'no data')
        ip_city = data.get('ip_city', 'no data')
        client = MongoClient('mongodb://localhost:27017/')
        db = client.home
        collection = db.mine
        count = collection.count_documents({"username": username,"password": password})
        if count==0:
            msg='账号或密码错误'
            return render_template("log_in.html", msg=msg)
        else:

            condition = {"username": username}
            # 找到信息
            user = collection.find_one(condition)
            # 更新后的信息
            user['ip_pro'] = ip_pro
            user['ip_city']=ip_city
            result = collection.update(condition, user)
            return redirect(url_for('search2',guest=username))
    return render_template("log_in.html")

@app.route('/getData', methods=['POST'])
def getData():
    page = int(request.form.get("currentPage"))
    limit = int(request.form.get("pageSize"))
    """
    search = request.form.get("key")
    from query import Query
    q = Query('../index_')
    res = q.standard_search(search)"""
    res = globals()["res"]
    print(page)
    start = (page - 1) * limit
    end = page * limit if len(res) > page * limit else len(res)
    ret = [{"POLICY_TITLE": res[i].get("POLICY_TITLE"),
            "POLICY_GRADE": res[i].get("POLICY_GRADE"),
            "PUB_TIME": res[i].get("PUB_TIME"),
            "POLICY_ID": res[i].get("POLICY_ID"),
            "PUB_AGENCY": res[i].get("PUB_AGENCY"),
            "PUB_AGENCY_ID": res[i].get("PUB_AGENCY_ID"),
            "PUB_AGENCY_FULLNAME": res[i].get("PUB_AGENCY_FULLNAME"),
            "PUB_NUMBER": res[i].get("PUB_NUMBER"),
            "POLICY_TYPE": res[i].get("POLICY_TYPE"),
            "UPDATE_DATE": res[i].get("UPDATE_DATE"),
            "POLICY_SOURCE": res[i].get("POLICY_SOURCE"),
            "POLICY_BODY": res[i].get("POLICY_BODY")} for i in range(start, end)]
    return {"data": ret, "count": len(res)}

@app.route('/search', methods=['GET'])
def search1():
    res = None
    msg='登录'
    search = request.args.get("q")
    from data.query import Query
    q = Query('../data/index_')
    res = q.standard_search(search)
    globals()["res"] = res
    """
        c = {
            'query': request.GET['q'],
            'resAmount': len(res),
            'results': res,
        }"""
    querys=search
    resAmount_ = len(res)
    results_ = res

    return render_template('result.html', query=querys,results=results_,resAmount=resAmount_,msg=msg)




@app.route('/guest/<guest>/search', methods=['GET'])
def search2(guest):
    res = None
    msg = "尊敬的" + str(guest) + "，您好"
    search = request.args.get("q")
    from data.query import Query
    q = Query('../data/index_')
    res = q.standard_search(str(search))
    globals()["res"] = res
    """
        c = {
            'query': request.GET['q'],
            'resAmount': len(res),
            'results': res,
        }"""
    querys = search
    resAmount_ = len(res)
    results_ = res

    return render_template('result.html', query=querys, results=results_, resAmount=resAmount_, msg=msg)
@app.route('/details',methods=['POST','GET'])
def detail():
    POLICY_ID=request.form.get('POLICY_ID')
    POLICY_GRADE=request.form.get('POLICY_GRADE')
    PUB_AGENCY_ID=request.form.get('PUB_AGENCY_ID')
    PUB_AGENCY=request.form.get('PUB_AGENCY')
    PUB_AGENCY_FULLNAME=request.form.get('PUB_AGENCY_FULLNAME')
    PUB_NUMBER=request.form.get('PUB_NUMBER')
    PUB_TIME=request.form.get('PUB_TIME')
    POLICY_TYPE=request.form.get('POLICY_TYPE')
    POLICY_TITLE=request.form.get('POLICY_TITLE')
    POLICY_BODY=request.form.get('POLICY_BODY')
    POLICY_SOURCE=request.form.get('POLICY_SOURCE')
    UPDATE_DATE=request.form.get('UPDATE_DATE')
    return render_template('detailed information.html',POLICY_ID=POLICY_ID,
                           POLICY_GRADE=POLICY_GRADE,PUB_AGENCY_ID=PUB_AGENCY_ID,
                           PUB_AGENCY=PUB_AGENCY,PUB_AGENCY_FULLNAME=PUB_AGENCY_FULLNAME,
                           PUB_NUMBER=PUB_NUMBER,PUB_TIME=PUB_TIME,
                           POLICY_TYPE=POLICY_TYPE,POLICY_TITLE=POLICY_TITLE,
                           POLICY_BODY=POLICY_BODY,POLICY_SOURCE=POLICY_SOURCE,
                           UPDATE_DATE=UPDATE_DATE)

from wsgiref.simple_server import make_server
if __name__ == '__main__':
    server = make_server('', 8000, app)
    server.serve_forever()