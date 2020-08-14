import os
import numpy as np
import random
import progressbar
from gensim import models
import tensorflow as tf
import matplotlib.pyplot as plt
import news_util
import single_softmax
import cnn_multi
import gru_attention
import stas

# News = news_util.news_c("news_len_64_dim_64")
News = news_util.news_c("news_s8_w8")
vocab_data = models.word2vec.Word2Vec.load("w64rev").wv.vectors
word_nums = vocab_data.shape[0]
word_dims = vocab_data.shape[1]
word_dims = 64
text_length = 64
doc_length = 8
sen_length = 8
cata_nums = 10
filter_nums = 64
filter_size = 1
filter_sizes = [1, 2, 3]
gru_units = 64
attention_size = 128
epochs = 10
batch_nums = 800
batch_size = 64


def main():

    # model = single_softmax.model(
    #     vocab_data, word_nums, word_dims, text_length, cata_nums)
    # model = cnn_multi.model(vocab_data, word_nums, word_dims,
    #                         text_length, cata_nums, filter_nums, filter_sizes)
    model = gru_attention.model(cata_nums=cata_nums, doc_length=doc_length,
                                sen_length=sen_length, vocab_data=vocab_data,
                                word_nums=word_nums, word_dims=word_dims,
                                gru_units=gru_units, attention_size=attention_size)

    train_data, train_target = News.data[:51200], News.target[:51200]
    valid_data, valid_target = News.data[-8800:-8000], News.target[-8800:-8000]
    test_data, test_target = News.data[-8000:], News.target[-8000:]
    cnt = 0
    recorder = stas.Stas(name="model_name")
    max_valid_pred = 0
    max_test_pred = 0
    max_train_pred = 0

    for epoch in range(epochs):

        pb = progressbar.ProgressBar(max_value=batch_nums)
        for i in range(batch_nums):
            data = train_data[i*batch_size:(i+1)*batch_size]
            target = train_target[i*batch_size:(i+1)*batch_size]
            loss = model.train(data, target)
            valid_pred = model.valid(valid_data, valid_target)
            recorder.update(cnt, Loss=loss, Valid_pred=valid_pred)
            if(valid_pred > max_valid_pred):
                max_valid_pred = valid_pred
                test_pred = model.test(test_data, test_target)
                recorder.update(cnt, Test_pred=test_pred)
                if test_pred > max_test_pred:
                    max_test_pred = test_pred
                    if test_pred > 0.91:
                        model.save()
            pb.update(i+1)
            cnt += 1

        train_pred = model.test(train_data, train_target)
        max_train_pred = max(max_train_pred, train_pred)
        recorder.update(cnt, Train_pred=train_pred)

        print("\nEpoch_{}:Max_test_pred_now = {}".format(epoch, max_test_pred))

        dt = list(zip(train_data, train_target))
        random.shuffle(dt)
        train_data[:], train_target[:] = zip(*dt)


if __name__ == '__main__':
    main()
