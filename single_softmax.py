import tensorflow as tf


class model:
    def __init__(self, vocab_data, word_nums, word_dims, text_length, cata_nums):
        self.G_S_S = tf.Graph()
        with self.G_S_S.as_default():
            self.X = tf.placeholder(tf.int32, [None, text_length])
            self.Y = tf.placeholder(tf.float32, [None, cata_nums])
            self.V = tf.Variable(vocab_data, expected_shape=[
                word_nums, word_dims])
            self.E = tf.nn.embedding_lookup(self.V, self.X)
            self.AE = tf.reduce_mean(self.E, reduction_indices=1)
            self.NORM = tf.layers.batch_normalization(self.AE)
            self.Y_ = tf.layers.dense(
                self.NORM, cata_nums,
                kernel_regularizer=tf.nn.l2_loss,
                bias_regularizer=tf.nn.l2_loss,)
            self.TOP_K = tf.nn.top_k(tf.nn.softmax(self.Y_), 3)
            self.CE = tf.nn.softmax_cross_entropy_with_logits(
                logits=self.Y_, labels=self.Y)

            self.LOSS = tf.reduce_mean(self.CE)
            self.STEP = tf.train.AdamOptimizer().minimize(self.LOSS)
            self.CORR = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
            self.PRED = tf.reduce_mean(tf.cast(self.CORR, tf.float32))
            self.SESS = tf.Session()
            self.SESS.run(tf.global_variables_initializer())
            self.SAVER = tf.train.Saver()
            self.MODEL_PATH = "./single_softmax_model"

    def train(self, data, target):
        loss, _ = self.SESS.run([self.LOSS, self.STEP], {
            self.X: data, self.Y: target})
        return loss

    def valid(self, data, target):
        return self.SESS.run(self.PRED, {self.X: data, self.Y: target})

    def test(self, data, target):
        bs = 100
        bn = (int)(len(data)/bs)
        correct = 0
        for i in range(bn):
            correct += self.SESS.run(self.PRED,
                                     {self.X: data[i*bs:i*bs+bs], self.Y: target[i*bs:i*bs+bs]})
        return correct/bn

    def predict(self, doc):
        return self.SESS.run(self.TOP_K, {self.X: [doc]})

    def save(self):
        self.SAVER.save(self.SESS, self.MODEL_PATH)

    def load(self):
        self.SAVER.restore(self.SESS, self.MODEL_PATH)
