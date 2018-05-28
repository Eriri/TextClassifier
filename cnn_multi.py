import tensorflow as tf


class model:
    def __init__(self, vocab_data, word_nums,
                 word_dims, text_length, cata_nums,
                 filter_nums, filter_sizes):

        self.X = tf.placeholder(tf.int32, [None, text_length])
        self.Y = tf.placeholder(tf.float32, [None, cata_nums])
        self.DROP_PROB = tf.placeholder(tf.float32)
        self.V = tf.Variable(vocab_data,
                             expected_shape=[word_nums, word_dims])
        self.E = tf.nn.embedding_lookup(self.V, self.X)
        pools = []
        for filter_size in filter_sizes:
            with tf.name_scope("conv_%s" % filter_size):
                self.CONV = tf.layers.conv1d(
                    self.E, filters=filter_nums, kernel_size=filter_size)
                self.NORM = tf.layers.batch_normalization(self.CONV)
                self.MAXPOOL = tf.reduce_max(self.CONV, reduction_indices=1)
                pools.append(self.MAXPOOL)
        self.MAXPOOL = tf.concat(pools, -1)
        self.H = tf.layers.dense(self.MAXPOOL, 128)
        self.D = tf.layers.dropout(self.H, rate=self.DROP_PROB)
        self.R = tf.nn.relu(self.D)
        self.Y_ = tf.layers.dense(self.R, cata_nums)
        self.Y_ = tf.reshape(self.Y_, shape=[-1, cata_nums])
        self.CE = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.Y_, labels=self.Y)

        self.LOSS = tf.reduce_mean(self.CE)
        self.STEP = tf.train.AdamOptimizer().minimize(self.LOSS)
        self.CORR = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
        self.PRED = tf.reduce_mean(tf.cast(self.CORR, tf.float32))
        self.SESS = tf.InteractiveSession()
        self.SESS.run(tf.global_variables_initializer())
        self.SAVER = tf.train.Saver()
        self.MODEL_PATH = "./cnn_multi_model"

    def train(self, data, target):
        loss, _ = self.SESS.run([self.LOSS, self.STEP], {self.X: data,
                                                         self.Y: target, self.DROP_PROB: 0.3})
        return loss

    def valid(self, data, target):
        return self.SESS.run(self.PRED, {self.X: data, self.Y: target, self.DROP_PROB: 0.0})

    def test(self, data, target):
        bs = 100
        bn = (int)(len(data)/bs)
        correct = 0
        for i in range(bn):
            correct += self.SESS.run(self.PRED,
                                     {self.X: data[i*bs:i*bs+bs], self.Y: target[i*bs:i*bs+bs], self.DROP_PROB: 0.0})
        return correct/bn

    def save(self):
        self.SAVER.save(self.SESS, self.MODEL_PATH)

    def load(self):
        self.SAVER.restore(self.SESS, self.MODEL_PATH)
