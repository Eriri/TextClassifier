import tensorflow as tf
import numpy as np
from gensim import models
import read_data
import progressbar


class CNN:
    def __init__(self, text_length, cata_nums, vocab_data, word_dim, filter_sizes):
        self.X = tf.placeholder(tf.int32, [None, text_length])
        self.Y = tf.placeholder(tf.float32, [None, cata_nums])
        self.drop_out_prob = tf.placeholder(tf.float32)
        self.Vocab = tf.constant(vocab_data, tf.float32, vocab_data.shape)
        self.Vocab_r = tf.variable(tf.random_uniform(
            vocab_data.shape, tf.float32, -1.0, 1.0))
        with tf.name_scope("embedding"):
            self.Embedding = tf.stack(
                [tf.nn.embedding_lookup(self.Vocab, self.X),
                 tf.nn.embedding_lookup(self.Vocab_r, self.X)],
                -1)
        for i, filter_size = enumerate(filter_sizes):
            with tf.name_scope("Convolution %s" % filter_size):
