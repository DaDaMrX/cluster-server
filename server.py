from bottle import post, request, run
import json

@post('/cluster')
def hello():
    username = request.forms.get('username')
    data = {'msg': username}
    return json.dumps(data)

run(host='localhost', port=8080, debug=True)