from gensim import models
import numpy as np
import re
import news_util

news = news_util.news(filename="data")
text_length = 512
cata_nums = 10


def generate_w2v():
    for i, text in enumerate(news.data):
        news.data[i] = text.split()
    w2v = models.word2vec.Word2Vec(news.data, size=64, min_count=1)
    w2v.save("w64rev")


def transform(text, cata, w2v):
    word_id = np.zeros(text_length, np.int32)
    text = text.split()
    raw_len = len(text)
    for i in range(min(raw_len, text_length)):
        word_id[i] = w2v.wv.vocab[text[i]].index
    if len(text) < text_length:
        for i in range(raw_len, text_length):
            word_id[i] = word_id[i % raw_len]
    cata_id = np.zeros(cata_nums, np.float32)
    cata_id[news.names.index(cata)] = 1.0
    return word_id, cata_id


def raw_data_trans():
    w2v = models.word2vec.Word2Vec.load("w64rev")
    for i in range(len(news.data)):
        news.data[i], news.target[i] = transform(
            news.data[i], news.target[i], w2v)
    news.data = np.array(news.data)
    news.target = np.array(news.target)
    np.savez("news_len_256_dim_64", data=news.data, target=news.target)


character = dict()
cnt = 0


def to_charac(text, cata):
    text = ('').join(text.split())
    global cnt
    word_id = np.zeros(text_length, np.int32)
    for i in range(min(text_length, len(text))):
        if text[i] not in character:
            character[text[i]] = cnt
            cnt += 1
        word_id[i] = character[text[i]]
    if len(text) < text_length:
        for i in range(len(text), text_length):
            word_id[i] = word_id[i % len(text)]
    cata_id = np.zeros(cata_nums, np.float32)
    cata_id[news.names.index(cata)] = 1.0
    return word_id, cata_id


def charac_data_trans():
    for i in range(len(news.data)):
        news.data[i], news.target[i] = to_charac(
            news.data[i], news.target[i])
    news.data = np.array(news.data)
    news.target = np.array(news.target)
    np.savez("news_charac_512", data=news.data,
             target=news.target, word_nums=cnt)


def att_trans(text, cata, w2v):
    text = re.split('。|！|？', text)
    text_id = []
    for sen in text:
        sen = sen.split()
        sen = [w2v.wv.vocab[w].index for w in sen if w in w2v.wv.vocab]
        slen = int(len(sen)/8)
        if len(sen) % 8 != 0:
            slen += 1
        for i in range(slen):
            s = np.zeros(8, np.int32)
            for j in range(i*8, min(i*8+8, len(sen))):
                s[j % 8] = sen[j]
            if i*8+8 > len(sen):
                for j in range(len(sen) % 8, 8):
                    s[j] = s[j % (len(sen) % 8)]
            text_id.append(s)
            if len(text_id) == 8:
                break
        if len(text_id) == 8:
            break
    if len(text_id) < 8:
        i = 0
        while len(text_id) < 8:
            text_id.append(text_id[i])
            i += 1
    text_id = np.array(text_id)
    cata_id = np.zeros(cata_nums, np.float32)
    cata_id[news.names.index(cata)] = 1.0
    return text_id, cata_id


def attention_raw_data():
    w2v = models.word2vec.Word2Vec.load("w64rev")
    for i in range(len(news.data)):
        news.data[i], news.target[i] = att_trans(
            news.data[i], news.target[i], w2v)
    news.data = np.array(news.data)
    news.target = np.array(news.target)
    np.savez("news_s8_w8", data=news.data, target=news.target)


def main():
    # generate_w2v()
    # raw_data_trans()
    # charac_data_trans()
    attention_raw_data()


if __name__ == '__main__':
    main()
