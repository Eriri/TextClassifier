import os
import numpy as np
import progressbar
from gensim import models
import tensorflow as tf
import matplotlib.pyplot as plt
import news_util
import single_softmax
import cnn_multi
import draw
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'

News = news_util.news_c("news_len_64_dim_64")
vocab_data = models.word2vec.Word2Vec.load("w64rev").wv.vectors
word_nums = vocab_data.shape[0]
word_dims = vocab_data.shape[1]
word_dims = 64
text_length = 64
cata_nums = 10
filter_nums = 64
filter_size = 1
filter_sizes = [1, 2, 3]
epochs = 20
batch_nums = 800
batch_size = 64
drawer = draw.drawer()


def main():

    s_s = single_softmax.model(
        vocab_data, word_nums, word_dims, text_length, cata_nums)
    # c_m = cnn_multi.model(vocab_data, word_nums, word_dims,
    #   text_length, cata_nums, filter_nums, filter_sizes)

    model = s_s

    valid_data, valid_target = News.data[-8800:-8000], News.target[-8800:-8000]
    test_data, test_target = News.data[-8000:], News.target[-8000:]
    cnt = 0
    max_valid_pred = 0
    max_test_pred = 0
    Loss_x = []
    Loss_y = []
    Valid_Pred_x = []
    Valid_Pred_y = []
    Test_Pred_x = []
    Test_Pred_y = []

    for epoch in range(epochs):
        pb = progressbar.ProgressBar(max_value=batch_nums)
        for i in range(batch_nums):
            data = News.data[i*batch_size:(i+1)*batch_size]
            target = News.target[i*batch_size:(i+1)*batch_size]
            loss = model.train(data, target)
            Loss_x.append(cnt)
            Loss_y.append(loss)
            valid_pred = model.valid(valid_data, valid_target)
            Valid_Pred_x.append(cnt)
            Valid_Pred_y.append(valid_pred)
            if(valid_pred > max_valid_pred):
                max_valid_pred = valid_pred
                test_pred = model.test(test_data, test_target)
                if test_pred > max_test_pred:
                    max_test_pred = test_pred
                    if test_pred > 0.91:
                        model.save()
                Test_Pred_x.append(cnt)
                Test_Pred_y.append(test_pred)
            pb.update(i+1)
            cnt += 1
        drawer.draw("Loss", "batches", Loss_x, "loss", Loss_y)
        drawer.draw("Valid_Pred", "batches",
                    Valid_Pred_x, "accu", Valid_Pred_y)
        drawer.draw("Test_Pred", "batches", Test_Pred_x, "accu", Test_Pred_y)
        print("\nMax_test_pred_now = {}".format(max_test_pred))


if __name__ == '__main__':
    main()
