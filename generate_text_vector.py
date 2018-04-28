import jieba
import read_data
import random
import numpy as np
from gensim import models

ds = read_data.Data_Set()
w2v = models.Word2Vec.load('../w2vmodel')
vocab = set(w2v.wv.vocab)
labels = list(set(ds.train_target))


def tf_text(text):
    text_vec = np.zeros(100)
    count = 1
    for word in jieba.cut(text):
        if word in vocab:
            text_vec += w2v.wv[word]
            count += 1
    text_vec /= count
    return text_vec


def tf_label(label):
    label_vec = np.zeros(len(labels))
    label_vec[labels.index(label)] = 1
    return label_vec


def generate_text_vector():
    for i in range(len(ds.train_data)):
        ds.train_data[i] = tf_text(ds.train_data[i])
        ds.train_target[i] = tf_label(ds.train_target[i])
    train = list(zip(ds.train_data, ds.train_target))
    random.shuffle(train)
    ds.train_data[:], ds.train_target[:] = zip(*train)

    for i in range(len(ds.valid_data)):
        ds.valid_data[i] = tf_text(ds.valid_data[i])
        ds.valid_target[i] = tf_label(ds.valid_target[i])

    for i in range(len(ds.test_data)):
        ds.test_data[i] = tf_text(ds.test_data[i])
        ds.test_target[i] = tf_label(ds.test_target[i])

    ds.train_data = np.array(ds.train_data)
    ds.train_target = np.array(ds.train_target)
    ds.valid_data = np.array(ds.valid_data)
    ds.valid_target = np.array(ds.valid_target)
    ds.test_data = np.array(ds.test_data)
    ds.test_target = np.array(ds.test_target)

    np.savez('../tvdata',
             train_data=ds.train_data, train_target=ds.train_target,
             valid_data=ds.valid_data, valid_target=ds.valid_target,
             test_data=ds.test_data, test_target=ds.test_target)


if __name__ == '__main__':
    generate_text_vector()
