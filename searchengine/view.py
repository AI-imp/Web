from django.shortcuts import render
import sys

sys.path.append('..')
from data.query import Query

q = Query('../十四届服务创新外包大赛数据/index_')

def search_form(request):
    return render(request, 'main.html')

def search_details(request):
    return render(request,'detailed information.html')

def search(request):
    res = None
    if 'q' in request.GET and request.GET['q']:
        res = q.standard_search(request.GET['q'])
        c = {
            'query': request.GET['q'],
            'resAmount': len(res),
            'results': res,
        }
    else:
        return render(request, 'main.html')

    # str = ''
    # for i in res:
    #     str += '<p><a href="' + i['newsUrl'] + '">' + i['newsTitle'] + '</a></p>'
    # return HttpResponse(str)

    return render(request, 'result.html', c)

