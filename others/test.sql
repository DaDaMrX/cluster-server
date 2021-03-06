-- SELECT
select * from CLUSTERRING_UTTERANCES;
select * from CLUSTERRING_TASKS;
select * from CLUSTERRING_convs;

INSERT INTO CLUSTERRING_CONVS (CONV_ID, BOT_ID, CLUSTER_ID) VALUES (0, '0', 0)
ON DUPLICATE KEY UPDATE
CONV_ID = VALUES(CONV_ID),
BOT_ID = VALUES(BOT_ID),
CLUSTER_ID = VALUES(CLUSTER_ID);

-- Reset
delete from CLUSTERRING_UTTERANCES;
delete from CLUSTERRING_convs;
delete from words;
delete from clusters;
delete from cluster_keywords;

UPDATE BOTS SET IS_COMPLETED = 0, PROGRESS = 0.0 WHERE BOT_ID = '0';
DELETE FROM CONVS WHERE BOT_ID = '0';
DELETE FROM WORDS WHERE BOT_ID = '0';
DELETE FROM CLUSTERS WHERE BOT_ID = '0';
DELETE FROM CLUSTER_KEYWORDS WHERE BOT_ID = '0';

-- BOTS 
select * from bots;
INSERT INTO BOTS (BOT_ID) VALUES ('0');
delete from bots;
UPDATE BOTS SET IS_COMPLETED = 1, PROGRESS = 1.0 WHERE BOT_ID = '0';

-- CLUSTERRING_HISTORY
select * from CLUSTERRING_HISTORY;
delete from CLUSTERRING_HISTORY;

-- UTTERANCES
select * from UTTERANCES;
SELECT DISTINCT CONV_ID FROM UTTERANCES WHERE BOT_ID = '0';
SELECT UTTERANCE FROM UTTERANCES WHERE BOT_ID = '0' AND CONV_ID = 0;

-- CONVS
select * from convs;
delete from convs;
SELECT CONVS.CONV_ID, SPEAKER_ID, UTTERANCE
FROM CONVS INNER JOIN UTTERANCES
WHERE CONVS.BOT_ID = '0' AND UTTERANCES.BOT_ID = CONVS.BOT_ID
AND CONVS.CONV_ID = UTTERANCES.CONV_ID
AND CLUSTER_ID = 0;

-- WORDS
select * from words;
INSERT INTO WORDS (WORD_ID, BOT_ID, WORD, DF) VALUES (0, '0', 'HELLO', 0.2);
delete from words;
SELECT DF FROM WORDS WHERE BOT_ID = '0' ORDER BY DF ASC;

-- CLUSTERS
select * from clusters;
INSERT INTO CLUSTERS (CLUSTER_ID, BOT_ID, VOLUME) VALUES (0, '0', 10), (1, '0', 20);
delete from clusters;

-- CLUSTER_KEYWORDS
select * from cluster_keywords;
delete from cluster_keywords;


