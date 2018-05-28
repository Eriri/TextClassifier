import news_util
import cnn_multi
import single_softmax
import gru_attention
from gensim import models

News = news_util.news_c("news_len_64_dim_64")
News_c = news_util.news_c("news_s8_w8")
vocab_data = models.word2vec.Word2Vec.load("w64rev").wv.vectors
word_nums = vocab_data.shape[0]
word_dims = vocab_data.shape[1]
word_dims = 64
text_length = 64
cata_nums = 10
filter_nums = 64
filter_size = 1
filter_sizes = [1, 2, 3]
doc_length = 8
sen_length = 8
gru_units = 64
attention_size = 128
epochs = 20
batch_nums = 800
batch_size = 64


# c_m = cnn_multi.model(vocab_data, word_nums, word_dims,
#                       text_length, cata_nums, filter_nums, filter_sizes)
# c_m.load()
# print(c_m.test(News.data[:51200], News.target[:51200]))

# s_s = single_softmax.model(
#     vocab_data, word_nums, word_dims, text_length, cata_nums)
# s_s.load()
# print(s_s.test(News.data[:51200], News.target[:51200]))

g_a = gru_attention.model(cata_nums=cata_nums, doc_length=doc_length,
                          sen_length=sen_length, vocab_data=vocab_data,
                          word_nums=word_nums, word_dims=word_dims,
                          gru_units=gru_units, attention_size=attention_size)
g_a.load()
print(g_a.test(News_c.data[:51200], News_c.target[:51200]))
