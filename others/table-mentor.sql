CREATE TABLE CLUSTERRING_TASKS(
    BOT_ID VARCHAR(32) PRIMARY KEY,
    IS_COMPLETED INT DEFAULT 0,
    PROGRESS DECIMAL(13, 10) NULL,
	CH DECIMAL(13, 10) NULL,
    TS TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE CLUSTERRING_HISTORY(
    BOT_ID VARCHAR(32) NOT NULL,
    STOPWORDS VARCHAR(10240), // WORD_ID的数组
    MIN_DF DECIMAL(11, 10) NULL,
    MAX_DF DECIMAL(11, 10) NULL,
    COUNT_CLUSTERS INT NOT NULL,
    METHOD VARCHAR(32),
    TS TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP

);

CREATE TABLE CLUSTERRING_STOPWORDS(
    BOT_ID VARCHAR(32) PRIMARY KEY,
    WORD_ID INT,
    TS TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE CLUSTERRING_UTTERANCE(
	UTTERANCE_ID INT PRIMARY KEY AUTO_INCREMENT,
    BOT_ID VARCHAR(32) NOT NULL,
	CONV_ID INT NOT NULL,
    SPEAKER_ID INT NOT NULL,
    UTTERANCE VARCHAR(10240) NOT NULL,
    TS TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE CLUSTERRING_CONVS (
    CONV_ID INT PRIMARY KEY,
    BOT_ID VARCHAR(32) NOT NULL,
    CLUSTER_ID INT NULL,
    TS TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE CLUSTERRING_WORDS (
    WORD_ID INT PRIMARY KEY,
    BOT_ID VARCHAR(32) NOT NULL,
    WORD VARCHAR(128) NOT NULL,
    DF DECIMAL(11, 10) NULL,
    TS TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE CLUSTERRING_CLUSTERS(
	CLUSTER_ID INT PRIMARY KEY,
    BOT_ID VARCHAR(32) NOT NULL,
    VOLUME INT NOT NULL,
    TS TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);

CREATE TABLE CLUSTERRING_CLUSTERKEYWORDS(
	ID INT PRIMARY KEY AUTO_INCREMENT,
    BOT_ID VARCHAR(32) NOT NULL,
    CLUSTER_ID INT NOT NULL,
    WORD_ID INT NOT NULL,
    TFIDF DECIMAL(20, 10) NOT NULL,
    TS TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
);
