import bottle
import json
import threading

from cluster import ClusterThread
from database import Database


class Server:
    
    def __init__(self, ip='127.0.0.1', port=8080, debug=False):
        self.ip = ip
        self.port = port
        self.debug = debug,

    @bottle.post('/')
    def cluster():
        data = bottle.request.json
        bot_id = data['bot_id']
        convs = data['convs']
        stop_words = data['stop_words']
        max_df = data['max_df']
        min_df = data['min_df']
        n_clusters = data['n_clusters']
        method = data['method']
        cluster_thread = ClusterThread(
            bot_id, convs, stop_words, max_df, min_df, n_clusters, method)
        cluster_thread.start()
        return 'successfull'
        
    def start(self):
        bottle.run(ip=self.ip, port=self.port, debug=self.debug)
        
        
Server().start()