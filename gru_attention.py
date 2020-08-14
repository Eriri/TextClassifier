import tensorflow as tf


class model:
    def __init__(self, cata_nums, doc_length, sen_length,
                 vocab_data, word_nums, word_dims,
                 gru_units, attention_size):
        self.G_G_A = tf.Graph()
        with self.G_G_A.as_default():
            self.gru_units = gru_units
            self.attention_size = attention_size

            self.X = tf.placeholder(tf.int32, [None, doc_length, sen_length])
            self.Y = tf.placeholder(tf.float32, [None, cata_nums])
            self.ikp = tf.placeholder(tf.float32)
            self.okp = tf.placeholder(tf.float32)

            self.V = tf.Variable(vocab_data, expected_shape=[
                word_nums, word_dims])
            self.E = tf.reshape(tf.nn.embedding_lookup(
                self.V, self.X), [-1, sen_length, word_dims])

            self.Ee = self.Encoder(self.E, "word")
            self.Een = tf.layers.batch_normalization(self.Ee)

            self.Ea = self.Attention(self.Een, "word")
            self.Ean = tf.layers.batch_normalization(self.Ea)

            self.S = tf.reshape(self.Ean, [-1, doc_length, 2*gru_units])

            self.Se = self.Encoder(self.S, "sentence")
            self.Sen = tf.layers.batch_normalization(self.Se)

            self.Sa = self.Attention(self.Sen, "sentence")
            self.San = tf.layers.batch_normalization(self.Sa)

            self.Doc = tf.reshape(self.San, [-1, 2*gru_units])

            self.Y_ = tf.layers.dense(self.Doc, cata_nums)
            self.TOP_K = tf.nn.top_k(tf.nn.softmax(self.Y_), 3)
            self.CE = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=self.Y, logits=self.Y_)
            self.LOSS = tf.reduce_mean(self.CE)
            self.STEP = tf.train.AdamOptimizer().minimize(self.LOSS)
            self.CORR = tf.equal(tf.argmax(self.Y, 1), tf.argmax(self.Y_, 1))
            self.PRED = tf.reduce_mean(tf.cast(self.CORR, tf.float32))
            self.SESS = tf.Session()
            self.SESS.run(tf.global_variables_initializer())
            self.SAVER = tf.train.Saver()
            self.MODEL_PATH = "./gru_attention_model"

    def Encoder(self, inputs, name_scope):
        with tf.name_scope(name_scope):
            fw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(num_units=self.gru_units, activation=tf.nn.tanh,
                                       reuse=tf.AUTO_REUSE, name=name_scope),
                input_keep_prob=self.ikp,
                output_keep_prob=self.okp)
            bw = tf.nn.rnn_cell.DropoutWrapper(
                tf.nn.rnn_cell.GRUCell(num_units=self.gru_units, activation=tf.nn.tanh,
                                       reuse=tf.AUTO_REUSE, name=name_scope),
                input_keep_prob=self.ikp,
                output_keep_prob=self.okp)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=fw, cell_bw=bw, inputs=inputs, dtype=tf.float32, scope=name_scope)
            final_outputs = tf.concat(outputs, 2)
            return final_outputs

    def Attention(self, inputs, name_scope):
        with tf.name_scope(name_scope):
            Uw = tf.Variable(tf.random_normal([self.attention_size]))
            Ut = tf.layers.dense(
                inputs=inputs, units=self.attention_size, activation=tf.nn.tanh)
            att = tf.expand_dims(tf.nn.softmax(
                tf.tensordot(Ut, Uw, [[2], [0]])), -1)
            out = tf.reduce_sum(tf.multiply(att, inputs), axis=1)
            return out

    def train(self, data, target):
        loss, _ = self.SESS.run(
            [self.LOSS, self.STEP], {self.X: data, self.Y: target, self.ikp: 0.7, self.okp: 0.7})
        return loss

    def valid(self, data, target):
        return self.SESS.run(self.PRED, {self.X: data, self.Y: target, self.ikp: 1.0, self.okp: 1.0})

    def test(self, data, target):
        bs = 100
        bn = (int)(len(data)/bs)
        correct = 0
        for i in range(bn):
            correct += self.SESS.run(self.PRED,
                                     {self.X: data[i*bs:i*bs+bs], self.Y: target[i*bs:i*bs+bs], self.ikp: 1.0, self.okp: 1.0})
        return correct/bn

    def predict(self, doc):
        return self.SESS.run(self.TOP_K, {self.X: [doc], self.ikp: 1.0, self.okp: 1.0})

    def save(self):
        self.SAVER.save(self.SESS, self.MODEL_PATH)

    def load(self):
        self.SAVER.restore(self.SESS, self.MODEL_PATH)
