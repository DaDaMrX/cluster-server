import urllib.parse, urllib.request
import json

from database import Database

url = 'http://localhost:8080'

data = {
    'bot_id': '0',
    'convs': [],
    'stop_words': [],
    'max_df': 0.1,
    'min_df': 20,
    'n_clusters': 50,
    'method': 'kmeans'
}

data = json.dumps(data).encode('utf8')
headers = {'content-type': 'application/json'}
req = urllib.request.Request(url, data=data, headers=headers)

result = urllib.request.urlopen(req).read().decode()
print(result)