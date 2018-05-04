import tensorflow as tf
import numpy as np
from gensim import models
import read_data
import progressbar


class CNN:
    def __init__(self, text_length, cata_nums, vocab_data, filter_size=3, filter_nums=128, drop_prob=0.5):
        self.X = tf.placeholder(tf.int32, [None, text_length], name="X")
        self.Y = tf.placeholder(tf.float32, [None, cata_nums], name="Y")
        self.drop_out_prob = tf.constant(
            drop_prob, tf.float32, name="drop_out_prob")
        self.Vocab = tf.constant(
            vocab_data, tf.float32, vocab_data.shape, name="Vocab")
        self.Vocab_random = tf.Variable(tf.random_uniform(
            vocab_data.shape, tf.float32, -1.0, 1.0), name="Vocab_random")
        self.Word_dim = tf.Variable(
            tf.int32, vocab_data.shape[1], name="Word_dim")
        with tf.name_scope("Embedding"):
            self.Embedding = tf.stack(
                [tf.nn.embedding_lookup(self.Vocab, self.X),
                 tf.nn.embedding_lookup(self.Vocab_random, self.X)],
                -1)
        with tf.name_scope("Convolution"):
            filter_shape = [filter_size, self.Word_dim, 2, filter_nums]
            strides = tf.constant([1, 1, 1, 1])
            w = tf.Variable(tf.truncated_normal(
                filter_shape, stddev=1.0), name="w")
            b = tf.Variable(tf.random_normal([filter_nums]), name="b")
            conv = tf.nn.conv2d(self.Embedding, w, strides,
                                "VALID", name="conv")
            h = tf.nn.relu(tf.nn.bias_add(conv, b))
            p = tf.reduce_max(h, reduction_indices=[1], name="p")
        with tf.name_scope("Output"):
            fc = tf.layers.dense(p, filter_nums)
            fc = tf.layers.dropout(fc, rate=self.drop_out_prob)
            fc = tf.nn.relu(fc)
            self.logits = tf.layers.dense(fc, cata_nums)
            self.Y_ = tf.argmax(tf.nn.softmax(self.logits), 1)
        with tf.name_scope("Loss"):
            self.CE = tf.nn.softmax_cross_entropy_with_logits(
                self.logits, self.Y)
            self.LOSS = tf.reduce_mean(self.CE)
            self.OPTIMIZER = tf.train.GradientDescentOptimizer(
                0.5).minimize(self.LOSS)
        with tf.name_scope("Accuracy"):
            correct = tf.equal(self.Y, tf.argmax(self.Y, 1))
            self.accu = tf.reduce_mean(tf.cast(correct, tf.float32))
