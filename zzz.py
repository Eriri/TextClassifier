import news_util
import re
import tensorflow as tf
from gensim import models

a = tf.Variable([[1], [2], [3], [4]])
c = tf.reshape(a, [-1, 2, 1])

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run(c))
