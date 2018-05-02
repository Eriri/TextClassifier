import read_data
import jieba
import random
import tensorflow as tf
import numpy as np
from gensim import models
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report


def softmax():
    tv = read_data.Text_Vector()
    X = tf.placeholder(tf.float32, [None, 100])
    W = tf.Variable(tf.zeros([100, 10]))
    B = tf.Variable(tf.zeros([10]))
    Y = tf.matmul(X, W)+B
    Y_ = tf.placeholder(tf.float32, [None, 10])
    CE = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y_, logits=Y))
    step = tf.train.GradientDescentOptimizer(0.5).minimize(CE)
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    batch_size = 50
    batch_steps = (int)(len(tv.train_data)/batch_size)
    for i in range(batch_steps):
        x = tv.train_data[i*batch_size:(i+1)*batch_size]
        y = tv.train_target[i*batch_size:(i+1)*batch_size]
        sess.run(step, feed_dict={X: x, Y_: y})
    CP = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
    ACCU = tf.reduce_mean(tf.cast(CP, tf.float32))
    print(sess.run(ACCU, {X: tv.test_data, Y_: tv.test_target}))


if __name__ == '__main__':
    softmax()
