import tensorflow as tf
import numpy as np
from gensim import models
import read_data
import progressbar

w2v = models.word2vec.Word2Vec.load('../w2vmodel')
vocab = set(w2v.wv.vocab)
pb = progressbar.ProgressBar(max_value=len(vocab))
T = []
i = 0
for word in vocab:
    T.append(w2v.wv[word])
    i += 1
    pb.update(i)
T = np.array(T)
np.savez('../embedding_word', embedding_word=T)
