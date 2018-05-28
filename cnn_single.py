import tensorflow as tf


class model:
    def __init__(self, vocab_data, word_nums, word_dims,
                 text_length, cata_nums,
                 filter_nums, fliter_size):
        self.X = tf.placeholder(tf.int32, [None, text_length])
        self.Y = tf.placeholder(tf.float32, [None, cata_nums])
        # with tf.device("/cpu:0"):
        #     self.V = tf.Variable(vocab_data, expected_shape=[
        #                          word_nums, word_dims])
        self.V = tf.get_variable(name="vocab_random", shape=[word_nums, word_dims],
                                 initializer=tf.random_normal_initializer())
        self.E = tf.nn.embedding_lookup(self.V, self.X)
        self.CONV1D = tf.layers.conv1d(
            self.E, filters=filter_nums, kernel_size=fliter_size)
        # self.NORM = tf.layers.batch_normalization(self.CONV1D)
        self.MAXPOOL = tf.reduce_max(self.CONV1D, reduction_indices=1)
        self.H = tf.layers.dense(self.MAXPOOL, 128, activation=tf.nn.relu)
        self.DROPOUT = tf.layers.dropout(self.H)
        self.R = tf.nn.relu(self.DROPOUT)
        self.Y_ = tf.layers.dense(self.R, cata_nums)
        self.CE = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=self.Y_, labels=self.Y)
        self.LOSS = tf.reduce_mean(self.CE)
        self.STEP = tf.train.AdamOptimizer().minimize(self.LOSS)
        self.CORR = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
        self.PRED = tf.reduce_mean(tf.cast(self.CORR, tf.float32))
        self.SESS = tf.InteractiveSession()
        self.SESS.run(tf.global_variables_initializer())
        self.WRITER = tf.summary.FileWriter(
            "./cnn_single_data128", self.SESS.graph)

    def train(self, data, target):
        loss, _ = self.SESS.run([self.LOSS, self.STEP], {
                                self.X: data, self.Y: target})
        correct = self.SESS.run(self.PRED, {self.X: data, self.Y: target})
        self.WRITER.add_summary(tf.Summary(
            value=[
                tf.Summary.Value(tag="loss", simple_value=loss),
                tf.Summary.Value(tag='train_accu', simple_value=correct)
            ]
        ))

    def test(self, data, target, tag_id):
        bn, bs = 100, 80
        correct = 0
        for i in range(bn):
            correct += self.SESS.run(self.PRED,
                                     {self.X: data[i*bs:i*bs+bs], self.Y: target[i*bs:i*bs+bs]})
        self.WRITER.add_summary(tf.Summary(
            value=[
                tf.Summary.Value(tag=tag_id, simple_value=correct)
            ]
        ))
        return correct/100
