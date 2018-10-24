import re
import jieba
import logging
import time
import threading

import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
%matplotlib inline


class ClusterThread(threading.Thread):
    
    def __init__(self, bot_id, convs, stop_words, max_df, min_df, n_clusters, method):
        threading.Thread.__init__(self)
        self.bot_id = bot_id
        self.convs = convs
        self.stop_words = stop_words
        self.max_df = max_df
        self.min_df = min_df
        self.n_clusters = n_clusters
        self.method = method
        self.max_features = 1000
        
        self.raw_file = 'chat-short-20w.txt'
        self.stop_wrods_file = 'stop_words.txt'

        jieba.setLogLevel(logging.INFO)
        logging.basicConfig(format='%(funcName)s:%(message)s', level=logging.DEBUG)
        
        self.corpus = []
        self.tfidf = None
        
        self.db_config = {
          'host': '127.0.0.1',
          'user': 'root',
          'password': 'xuhengda',
          'database': 'TEST',
        }
        self.db = Database(self.db_config)
        
    def run(self):
        self.db.new_bot(self.bot_id)
#         time.sleep(4)
        self.data_cleaning()
        self.convert_tfidf()
        logging.info('Finish.')
        self.db.update_progress(self.bot_id, 1, 1.0)
        
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
        
    def data_cleaning(self):
        logging.info('数据清洗开始...')
        self.corpus = self.db.select_all_convs(self.bot_id)

        # 正则表达式替换特定字符串
        self.corpus = list(map(self._substitute, self.corpus))
        logging.info('正则表达式替换完成.')

        # 分词
        t = time.time()
        self.corpus = list(map(jieba.cut, self.corpus))
        logging.info('分词完成.')

        # 删除停用词
        # TODO: 需要读数据库吗？
        logging.info('删除停用词开始...')
        with open(self.stop_wrods_file, encoding='utf-8') as f:
            stop_words = f.read().strip().split('\n')

        self.stop_words.extend(stop_words)

        t = time.time()
        for i in range(len(self.corpus)):
            tokens = []
            for token in self.corpus[i]:
                token = token.strip()
                if len(token) > 1 and token not in self.stop_words:
                    tokens.append(token)
            self.corpus[i] = tokens
        logging.info('删除停用词完成 (用时: %.2fs).' % (time.time() - t))

        # 组合
        self.corpus = list(map(lambda x: ' '.join(x), self.corpus))

        logging.info('数据清洗完成.')
                
    def convert_tfidf(self):
        tfidf_vectorizer = TfidfVectorizer(max_df=self.max_df,
                                           min_df=self.min_df,
                                           max_features=self.max_features)
        self.tfidf = tfidf_vectorizer.fit_transform(self.corpus)
        
        # 获取每个词的df值
        # TODO: 写数据库, tfidf: 2维, df: 1维
        idf = tfidf_vectorizer.idf_
        df = list(map(lambda x: (len(self.corpus) + 1) / np.exp(x - 1) - 1, idf))
        df = list(map(lambda x: x / len(self.corpus), df))
#         df.sort()
        
        terms = tfidf_vectorizer.get_feature_names()
        words = []
        for idx, word in enumerate(terms):
            words.append([idx, word])
            
        for i in range(len(words)):
            words[i].append(df[i])
        self.db.insert_words_with_df(self.bot_id, words)

        # TODO: 前端读数据库
        print('文件频率df-词数')
        plt.hist(df, 200)
        plt.show()

        print('每个词的文件频率df')
        plt.bar(np.arange(len(df)), df)
        plt.show()
        
        # 将每段对话中的词按tfidf值从高到低排序
        # TODO: 前端显示总词数
        print('total words:', len(tfidf_vectorizer.vocabulary_), end='\n\n')
        
        # 写数据库tfidf
        logging.info('开始写入tfidf值...')
        self.db.insert_tfidfs(self.bot_id, self.tfidf.toarray())
        logging.info('写入tfidf值完成.')

        # 打印指定对话中的关键词
        # TODO: 前端
        terms = tfidf_vectorizer.get_feature_names()
        conv_idx = 0
        for row in self.tfidf[:4]:
            conv_idx += 1
            print('Conv %d: ' % conv_idx, end='')
            row = row.toarray().squeeze()
            num = min(30, (row != 0).sum())
            indexes = row.argsort()[:-1-num:-1]
            words = [terms[idx] for idx in indexes]
            print(' '.join(words), end='\n\n')

            values = [row[idx] for idx in indexes]
            plt.bar(range(len(values)), values)
            plt.show()