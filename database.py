import mysql.connector


class Database:
    
    def __init__(self, config):
        self.config = config
        self.cnx = mysql.connector.connect(**config)
        
    def new_bot(self, bot_id):
        cursor = self.cnx.cursor()
        sql = 'INSERT INTO CLUSTERING_STATUS (BOT_ID) VALUES (%s)'
        data = (bot_id,)
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        
    def query_progress(self, bot_id):
        cursor = self.cnx.cursor()
        sql = 'SELECT IS_COMPLETED, PROGRESS FROM CLUSTERING_STATUS WHERE BOT_ID = %s'
        data = (bot_id,)
        cursor.execute(sql, data)
        is_completed = False
        progress = 0.0
        for c, p in cursor:
            is_completed = True if int(c) == 1 else False
            progress = float(p)
        cursor.close()
        return is_completed, progress
    
    def update_progress(self, bot_id, is_completed, progress):
        cursor = self.cnx.cursor()
        sql = ('UPDATE CLUSTERING_STATUS SET IS_COMPLETED = %s, PROGRESS = %s '
               'WHERE BOT_ID = %s')
        data = (is_completed, progress, bot_id)
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()

    def insert_conv(self, bot_id, conv_id, conv):
        cursor = self.cnx.cursor()
        sql = 'INSERT INTO CONVS (CONV_ID, BOT_ID, CONV) VALUES (%s, %s, %s)'
        data = (conv_id, bot_id, conv)
        cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        
    def insert_convs(self, bot_id, conversations):
        cursor = self.cnx.cursor()
        for conv_id, conv in conversations:
            sql = 'INSERT INTO CONVS (CONV_ID, BOT_ID, CONV) VALUES (%s, %s, %s)'
            data = (conv_id, bot_id, conv)
            cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        
    def select_all_convs(self, bot_id):
        cursor = self.cnx.cursor()
        sql = 'SELECT CONV FROM CONVS WHERE BOT_ID = %s'
        data = (bot_id,)
        cursor.execute(sql, data)
        corpus = [row[0] for row in cursor]
        cursor.close()
        return corpus
    
    def insert_word(self, bot_id, word_id, word):
        cursor = self.cnx.cursor()
        sql = ('INSERT INTO WORDS (WORD_ID, BOT_ID, WORD)'
               'VALUES (%s, %s, %s)')
        data = (word_id, bot_id, word)
        cursor.execute(sql)
        self.cnx.commit()
        cursor.close()
        
    def insert_words(self, bot_id, words):
        cursor = self.cnx.cursor()
        for word_id, word in words:
            sql = 'INSERT INTO WORDS (WORD_ID, BOT_ID, WORD) VALUES (%s, %s, %s)'
            data = (word_id, bot_id, word)
            cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        
    def insert_words_with_df(self, bot_id, words):
        cursor = self.cnx.cursor()
        for word_id, word, df in words:
            sql = 'INSERT INTO WORDS (WORD_ID, BOT_ID, WORD, DF) VALUES (%s, %s, %s, %s)'
            data = (word_id, bot_id, word, df)
            cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        
    def select_all_words(self, bot_id):
        cursor = self.cnx.cursor()
        sql = 'SELECT WORD FROM WORDS WHERE BOT_ID = %s'
        data = (bot_id,)
        cursor.execute(sql, data)
        words = [row[0] for row in cursor]
        cursor.close()
        return words
    
    def insert_dfs(self, bot_id, dfs):
        cursor = self.cnx.cursor()
        for word_id, df in dfs:
            sql = ('UPDATE WORDS SET DF = %s'
                   'WHERE BOT_ID = %s AND WORD_ID = %s')
            data = (df, bot_id, word_id)
            cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        
    def select_all_dfs(self, bot_id):
        cursor = self.cnx.cursor()
        sql = 'SELECT DF FROM WORDS WHERE BOT_ID = %s'
        data = (bot_id,)
        cursor.execute(sql, data)
        dfs = [float(row[0]) for row in cursor]
        cursor.close()
        return dfs
    
    def insert_tfidfs(self, bot_id, tfidfs):
        cursor = self.cnx.cursor()
        sql = 'INSERT INTO TFIDFS (TFIDF_ID, BOT_ID, CONV_ID, WORD_ID, TFIDF) VALUES'
        tfidf_id = 0
        for conv_id, row in enumerate(tfidfs):
            for word_id, tfidf in enumerate(row):
                sql = 'INSERT INTO TFIDFS (TFIDF_ID, BOT_ID, CONV_ID, WORD_ID, TFIDF) VALUES'
                sql += ' (%s, %s, %s, %s, %s)'
                data = (tfidf_id, bot_id, conv_id, word_id, tfidf)
                tfidf_id += 1
                cursor.execute(sql, data)
        self.cnx.commit()
        cursor.close()
        
    def select_all_tfidfs(self, bot_id):
        cursor = self.cnx.cursor()
        sql = 'SELECT CONV_ID, WORD_ID, TFIDF FROM TFIDFS WHERE BOT_ID = %s'
        data = (bot_id,)
        cursor.execute(sql, data)
        tfidfs = [[conv_id, word_id, float(tfidf)] for conv_id, word_id, tfidf in cursor]
        cursor.close()
        return tfidfs
        
    def __del__(self):
        self.cnx.close()
