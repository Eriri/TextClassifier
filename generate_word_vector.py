from gensim import models
import read_data
import jieba
import re
import progressbar


def generate_word_vector():
    ds = read_data.Data_Set()
    corpus = []
    endword = [' ', ',', '.', '?', '!', ';', '<', '>', '"', ':', '[', ']', '(', ')',
               '，', '。', '？', '！', '《', '》', '；', '：', '“', '”', '‘', '’', '【', '】', '『', '』', '（', '）']
    pb = progressbar.ProgressBar(max_value=14*2500)
    i = 1

    for text in ds.train_data:
        final_text = []
        for word in jieba.cut(text):
            if word in endword and len(final_text) > 0:
                corpus.append(final_text)
                final_text = []
            word = re.sub('[^\u4e00-\u9fa5]', '', word)
            if word != '' and word not in ds.stopwords:
                final_text.append(word)
        if len(final_text) > 0:
            corpus.append(final_text)
        pb.update(i)
        i += 1
    w2v = models.Word2Vec(sentences=corpus)
    w2v.save('w2vmodel')


if __name__ == '__main__':
    generate_word_vector()
