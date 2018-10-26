import pymysql.cursors
import bottle
import json
import threading
import jieba
import logging
import re
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
import os


jieba.setLogLevel(logging.INFO)
logging_format = '%(funcName)s:%(message)s'
logging.basicConfig(format=logging_format, level=logging.DEBUG)

db_config = {
    'host': os.environ['MYSQL_HOST'],
    'port': int(os.environ['MYSQL_PORT']), 
    'user': os.environ['MYSQL_USER'],
    'password': os.environ['MYSQL_PASSWORD'],
    'db': os.environ['MYSQL_DB']
}

# db_config = {
#     'host': '9.59.150.218',
#     'port': 3306, 
#     'user': 'iboost',
#     'password': 'passw0rd',
#     'db': 'BOOSTDB'
# }

# db_config = {
#     'host': '127.0.0.1',
#     'port': 3306, 
#     'user': 'root',
#     'password': 'xuhengda',
#     'db': 'BOOSTDB'
# }


class ClusterThread(threading.Thread):
    
    def __init__(self, bot_id, stop_words, max_df, min_df, n_clusters, method):
        threading.Thread.__init__(self)
        # self.db_config = {
        #     'host': '9.59.150.218',
        #     'port': 3306, 
        #     'user': 'iboost',
        #     'password': 'passw0rd',
        #     'db': 'BOOSTDB'
        # }
        # self.cnx = mysql.connector.connect(**self.db_config)
        self.cnx = pymysql.connect(**db_config)

        self.bot_id = bot_id
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.n_clusters = n_clusters
        self.method = method

        self.corpus = None
        self.max_features = 1000
        self.tfidf_vectorizer = None
        self.tfidf = None
        self.max_iter = 100
        self.labels = None

    def run(self):
        self.save_param()
        self.save_stopwords()
        self.read_data()
        self.clean_data()
        self.calc_tfidf()
        self.cluster()
        self.calc_ch_score()
        self.write_label()
        self.write_cluster_vol()
        self.calc_keywords()
        self.finish()
    
    def save_param(self):
        logging.info('Saving param begin...')
        cursor = self.cnx.cursor()
        sql = ('INSERT INTO CLUSTERRING_HISTORY '
            '(BOT_ID, MAX_DF, MIN_DF, STOPWORDS, N_CLUSTERS, METHOD) '
            'VALUES (%s, %s, %s, %s, %s, %s)')
        stopwords = ','.join(self.stop_words)
        data = (self.bot_id, self.max_df, self.min_df,
            stopwords, self.n_clusters, self.method)
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        logging.info('Saving param finish.')

    def save_stopwords(self):
        if not self.stop_words:
            return
        logging.info('Saving stopwords begin...')
        cursor = self.cnx.cursor()
        sql = 'INSERT INTO CLUSTERRING_STOPWORDS (BOT_ID, WORD) VALUES'
        sql_values = ' (%s, %s),' * len(self.stop_words)
        sql += sql_values[:-1]
        data = []
        for word in self.stop_words:
            data.append(self.bot_id)
            data.append(word)
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        logging.info('Saving stopwords finished.')

    def read_data(self):
        logging.info('Reading data begin...')
        cursor = self.cnx.cursor()
        sql = 'SELECT CONV_ID, UTTERANCE FROM CLUSTERRING_UTTERANCES WHERE BOT_ID = %s'
        data = (self.bot_id,)
        cursor.execute(sql, data)

        self.corpus = []
        conv = ''
        last_conv_id = None
        for conv_id, utterance in cursor:
            if conv_id != last_conv_id:
                self.corpus.append(conv[:len(conv) - 1])
                conv = ''
                last_conv_id = conv_id
            else:
                conv += utterance + ' '
        self.corpus.append(conv[:len(conv) - 1])
        self.corpus = self.corpus[1:]
        cursor.close()
        logging.info('%d conversations.' % len(self.corpus))

        cursor = self.cnx.cursor()
        sql = 'SELECT DISTINCT CONV_ID FROM CLUSTERRING_UTTERANCES WHERE BOT_ID = %s'
        data = (self.bot_id,)
        cursor.execute(sql, data)
        conv_ids = list(map(lambda t: int(t[0]), cursor))

        sql = 'INSERT INTO CLUSTERRING_CONVS (CONV_ID, BOT_ID) VALUES '
        sql_value = '(%s, %s), ' * len(conv_ids)
        sql += sql_value[:-2]
        data = []
        for conv_id in conv_ids:
            data.append(conv_id)
            data.append(self.bot_id)
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        logging.info('Reading data finished.')
        
    def _substitute(self, sent):
        exps = [
            r'#E-\w\[数字x\]|~O\(∩_∩\)O/~',
            r'http[s]?://[a-zA-Z0-9|\.|/]+',
            r'http[s]?://[a-zA-Z0-9\./-]*\[链接x\]',
            r'\[ORDERID_[0-9]+\]',
            r'\[日期x\]',
            r'\[时间x\]',
            r'\[金额x\]',
            r'\[站点x\]',
            r'\[数字x\]',
            r'\[地址x\]',
            r'\[姓名x\]',
            r'\[邮箱x\]',
            r'\[电话x\]',
            r'\[商品快照\]',
            r'<s>',
            r'\s+',
            r'[a-z|0-9]+'
            "[\s+\.\!\/_,$%^:*(+\"\')]+",
            "[+——()?:【】‘’“”`！，。？、~@#￥%……&*（）]+"
        ]
        for exp in exps:
            sent = re.sub(exp, ' ', sent)
        return sent
        
    def clean_data(self):
        logging.info('Cleaning data begin...')
        stop_wrods_file = 'stop_words.txt'

        # 正则表达式替换特定字符串
        self.corpus = list(map(self._substitute, self.corpus))
        logging.info('正则表达式替换完成.')

        # 分词
        t = time.time()
        self.corpus = list(map(jieba.cut, self.corpus))
        logging.info('分词完成.')

        # 删除停用词
        logging.info('删除停用词开始...')
        with open(stop_wrods_file, encoding='utf-8') as f:
            stop_words = f.read().strip().split('\n')
        stop_words.extend(self.stop_words)

        t = time.time()
        for i in range(len(self.corpus)):
            tokens = []
            for token in self.corpus[i]:
                token = token.strip()
                if len(token) > 1 and token not in stop_words:
                    tokens.append(token)
            self.corpus[i] = tokens
        logging.info('删除停用词完成 (用时: %.2fs).' % (time.time() - t))

        # 组合
        self.corpus = list(map(lambda x: ' '.join(x), self.corpus))
        logging.info('Cleaning data finished.')

    def calc_tfidf(self):
        logging.info('Calc tfidf begin...')
        self.tfidf_vectorizer = TfidfVectorizer(max_df=self.max_df,
                                   min_df=self.min_df,
                                   max_features=self.max_features)
        self.tfidf = self.tfidf_vectorizer.fit_transform(self.corpus)

        # 获取每个词的df值
        idf = self.tfidf_vectorizer.idf_
        df = list(map(lambda x: (len(self.corpus) + 1) / np.exp(x - 1) - 1, idf))
        df = list(map(lambda x: x / len(self.corpus), df))

        words = self.tfidf_vectorizer.get_feature_names()

        cursor = self.cnx.cursor()
        sql = 'INSERT INTO CLUSTERRING_WORDS (WORD_ID, BOT_ID, WORD, DF) VALUES '
        sql_values = '(%s, %s, %s, %s), ' * len(words)
        sql += sql_values[:-2]
        data = []
        for (word_id, (word, df)) in enumerate(zip(words, df)):
            data.extend([word_id, self.bot_id, word, float(df)])
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        logging.info('Calc tfidf finished.')

    def cluster(self):
        logging.info('Cluster begin...')
        kmeans = KMeans(n_clusters=self.n_clusters,
                    max_iter=self.max_iter,
                    n_init=8,
                    init='k-means++',
                    n_jobs=-1,
                    random_state=0,
                    verbose=1)
        self.labels = kmeans.fit_predict(self.tfidf)
        logging.info('Cluster begin...')

    def calc_ch_score(self):
        logging.info('Calc CH score begin...')
        ch = metrics.calinski_harabaz_score(self.tfidf.toarray(), self.labels)
        logging.info('CH: %.3f' % ch)

        cursor = self.cnx.cursor()
        sql = 'UPDATE CLUSTERRING_TASKS SET CH = %s WHERE BOT_ID = %s'
        data = (float(ch), self.bot_id)
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        logging.info('Calc CH score finished.')

    def write_label(self):
        logging.info('Writing label begin...')
        cursor = self.cnx.cursor()

        sql = 'INSERT INTO CLUSTERRING_CONVS (CONV_ID, BOT_ID, CLUSTER_ID) VALUES'
        sql_values = ' (%s, %s, %s),' * len(self.labels)
        sql_tail = (' ON DUPLICATE KEY UPDATE '
        'CONV_ID = VALUES(CONV_ID), '
        'BOT_ID = VALUES(BOT_ID), '
        'CLUSTER_ID = VALUES(CLUSTER_ID)')
        sql += sql_values[:-1] + sql_tail

        data = []
        for conv_id, cluster_id in enumerate(self.labels):
            data.extend([int(conv_id), self.bot_id, int(cluster_id), ])

        cursor.execute(sql, data)
        self.cnx.commit()
        logging.info('commited')
        cursor.close()
        logging.info('Writing label finished.')

    def write_cluster_vol(self):
        logging.info('Writing cluster vol begin...')
        counts = [(self.labels == idx).sum() for idx in range(self.n_clusters)]
        cursor = self.cnx.cursor()
        sql = 'INSERT INTO CLUSTERRING_CLUSTERS (CLUSTER_ID, BOT_ID, VOLUME) VALUES'
        sql_values = ' (%s, %s, %s),' * self.n_clusters
        sql += sql_values[:-1]
        data = []
        for cluster_id, volume in enumerate(counts):
            data.extend([cluster_id, self.bot_id, int(volume)])
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        logging.info('Writing cluster vol finished.')

    def calc_keywords(self):
        logging.info('Calc keywords begin...')
        cursor = self.cnx.cursor()
        td = list(map(lambda x: x.toarray().squeeze(), self.tfidf))
            
        vectorss = [[] for _ in range(self.n_clusters)]
        for i in range(len(self.corpus)):
            vectorss[self.labels[i]].append(td[i])
        terms = self.tfidf_vectorizer.get_feature_names()
            
        for cluster_id, vectors in enumerate(vectorss):
            vectors = np.array(vectors)
            vectors = vectors.sum(0)
            num = (vectors != 0.0).sum()
            word_ids = vectors.argsort()[:-1-num:-1]
            
            sql = 'INSERT INTO CLUSTERRING_CLUSTERKEYWORDS (BOT_ID, CLUSTER_ID, WORD_ID, TFIDF) VALUES'
            sql_values = ' (%s, %s, %s, %s),' * len(word_ids)
            sql += sql_values[:-1]
            data = []
            for word_id in word_ids:
                data.extend((self.bot_id, cluster_id, int(word_id), float(vectors[word_id])))
            cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        logging.info('Calc keywords finished.')

    def finish(self):
        logging.info('Finish begin...')
        cursor = self.cnx.cursor()
        sql = 'UPDATE CLUSTERRING_TASKS SET IS_COMPLETED = %s, PROGRESS = %s WHERE BOT_ID = %s'
        data = (1, 1.0, self.bot_id)
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        self.cnx.close()
        logging.info('Finish finished.')


class Server(bottle.Bottle):
    
    def __init__(self):
        super(Server, self).__init__()
        self.route('/cluster', method='POST', callback=self.cluster)
        # self.db_config = {
        #     'host': '9.59.150.218',
        #     'port': 3306, 
        #     'user': 'iboost',
        #     'password': 'passw0rd',
        #     'db': 'BOOSTDB'
        # }
        self.cnx = pymysql.connect(**db_config)
        
    def cluster(self):
        logging.info('received request')
        data = bottle.request.json
        if not self.check_bot_id(data['bot_id']):
            return '{"status": "no bot_id"}'
        self.reset_database(data['bot_id'])
        cluster_thread = ClusterThread(
            data['bot_id'],
            data['stop_words'],
            data['max_df'],
            data['min_df'],
            data['n_clusters'],
            data['method'])
        cluster_thread.start()
        return '{"status": "successful"}'

    def check_bot_id(self, bot_id):
        logging.info('check_bot_id...')
        # cnx = pymysql.connector.connect(**self.db_config)
        # cnx = pymysql.connect(**self.db_config)
        cursor = self.cnx.cursor()
        sql = 'SELECT BOT_ID FROM CLUSTERRING_TASKS WHERE BOT_ID = %s'
        data = (bot_id,)
        cursor.execute(sql, data)
        row = cursor.fetchone()
        result = True if row else False
        cursor.close()
        # cnx.close()
        logging.info('check_bot_id finish.')
        return result

    def reset_database(self, bot_id):
        logging.info('reset database...')
        # cnx = mysql.connector.connect(**self.db_config)
        # cnx = pymysql.connect(**self.db_config)
        cursor = self.cnx.cursor()
        sqls = [
            'UPDATE CLUSTERRING_TASKS SET IS_COMPLETED = 0, PROGRESS = 0.0 WHERE BOT_ID = %s',
            'DELETE FROM CLUSTERRING_HISTORY WHERE BOT_ID = %s',
            'DELETE FROM CLUSTERRING_STOPWORDS WHERE BOT_ID = %s',
            'DELETE FROM CLUSTERRING_CONVS WHERE BOT_ID = %s',
            'DELETE FROM CLUSTERRING_WORDS WHERE BOT_ID = %s',
            'DELETE FROM CLUSTERRING_CLUSTERS WHERE BOT_ID = %s',
            'DELETE FROM CLUSTERRING_CLUSTERKEYWORDS WHERE BOT_ID = %s',
        ]
        data = (bot_id,)
        for sql in sqls:
            logging.info('delete one...')        
            cursor.execute(sql, data)
            self.cnx.commit()
            logging.info('delete one')
        cursor.close()
        # cnx.close()
        logging.info('reset database finished.')
    

if __name__ == '__main__':
    Server().run(ip='127.0.0.0', port=8080, debug=True)
