import urllib.parse, urllib.request

url = 'http://localhost:8080/cluster'

data = {
    'username': 'DaDa',
}
data = urllib.parse.urlencode(data).encode()

result = urllib.request.urlopen(url, data).read().decode()
print(result)