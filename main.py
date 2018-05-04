import read_data
import jieba
from gensim import models
import tensorflow as tf
import numpy as np


v = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]], dtype=np.float32)
v = tf.constant(v, dtype=tf.float32)
x = tf.constant([0, 1, 2])
X = tf.nn.embedding_lookup(v, x)
X = tf.expand_dims(X, 0)
X = tf.expand_dims(X, -1)
w = np.array([[1., 0., 0.], [0., 0., 1.]], dtype=np.float32)
w = np.reshape(w, (1, 3, 1, 2))
w = tf.constant(w)
y = tf.nn.conv2d(X, w, [1, 1, 1, 1], "VALID")
p = tf.reduce_max(y, reduction_indices=[1])
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
print(sess.run(p))


# writer = tf.summary.FileWriter("D:/log", sess.graph)

# writer.close()
