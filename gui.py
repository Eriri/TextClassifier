import sys
import tkinter
import jieba
from gensim import models
import numpy as np
import re
from sklearn.externals import joblib
import cnn_multi
import single_softmax
import gru_attention
import news_util

cata_names = ['房产', '股票', '财经', '娱乐',
              '社会', '教育', '体育', '科技', '时政', '游戏']
w2v = models.word2vec.Word2Vec.load("w64rev")
vocab = set(w2v.wv.vocab)
vocab_data = w2v.wv.vectors
word_nums = vocab_data.shape[0]
word_dims = vocab_data.shape[1]
text_length = 64
cata_nums = 10
filter_nums = 64
filter_size = 1
filter_sizes = [1, 2, 3]
doc_length = 8
sen_length = 8
gru_units = 64
attention_size = 128

c_m = cnn_multi.model(vocab_data, word_nums, word_dims,
                      text_length, cata_nums, filter_nums, filter_sizes)
c_m.load()

s_s = single_softmax.model(vocab_data, word_nums,
                           word_dims, text_length, cata_nums)
s_s.load()

g_a = gru_attention.model(cata_nums, doc_length, sen_length,
                          vocab_data, word_nums, word_dims, gru_units, attention_size)
g_a.load()

SVM = joblib.load("svm_model")

MNB = joblib.load("mnb_model")

TV = joblib.load("TVmodel")
feature_names = TV.get_feature_names()

root = tkinter.Tk()
root.title("Text Classification")
root.geometry("720x500")

text = tkinter.Text(root, height=18, width=72)
text.place(x=5, y=5)

ss = ""
doc = []


def pre_excu():
    s = text.get(1.0, tkinter.END)
    text.delete(1.0, tkinter.END)
    doc.clear()
    global ss
    ss = (" ").join(jieba.cut(s))
    for w in jieba.cut(s):
        if w in vocab:
            doc.append(w)
    text.insert(tkinter.END, (" ").join(doc))
    text.config(state=tkinter.DISABLED)


def trans_doc():
    word_vec = np.zeros(text_length, np.int32)
    raw_len = len(doc)
    for i in range(min(raw_len, text_length)):
        word_vec[i] = w2v.wv.vocab[doc[i]].index
    if raw_len < text_length:
        for i in range(raw_len, text_length):
            word_vec[i] = word_vec[i % raw_len]
    return word_vec


def trans_doc_h():
    global ss
    ss = re.split('。|！|？|……', ss)
    word_vec = []
    for sen in ss:
        sen = sen.split()
        sen = [w2v.wv.vocab[w].index for w in sen if w in vocab]
        slen = int(len(sen)/8)
        if len(sen) % 8 != 0:
            slen += 1
        for i in range(slen):
            s = np.zeros(8, np.int32)
            for j in range(i*8, min(i*8+8, len(sen))):
                s[j % 8] = sen[j]
            if i*8+8 > len(sen):
                for j in range(len(sen) % 8, 8):
                    s[j] = s[j % (len(sen) % 8)]
            word_vec.append(s)
            if len(word_vec) == 8:
                break
        if len(word_vec) == 8:
            break
    if len(word_vec) < 8:
        i = 0
        while len(word_vec) < 8:
            word_vec.append(word_vec[i])
            i += 1
    word_vec = np.array(word_vec)
    return word_vec


def classify():
    print("classify")
    result = []
    svm["text"] = "SVM:%s" % SVM.predict([(" ").join(doc)])[0]
    mnb["text"] = "MNB:%s" % MNB.predict([(" ").join(doc)])[0]
    word_vec = trans_doc()
    word_vec_h = trans_doc_h()
    result.append(c_m.predict(word_vec))
    result.append(s_s.predict(word_vec))
    result.append(g_a.predict(word_vec_h))
    for i in range(3):
        for j in range(3):
            res[i][j]["text"] = str(result[i][0][0][j]*100)[:4]+"%"
            canvas[i][j].coords(cr[i][j], 1, 1, result[i][0][0][j]*100, 20)
            label[i][j]["text"] = cata_names[result[i][1][0][j]]


def clear():
    text.config(state=tkinter.NORMAL)
    text.delete(1.0, tkinter.END)
    text_key_word.delete(1.0, tkinter.END)
    for i in range(3):
        for j in range(3):
            label[i][j]["text"] = "top_%d" % (j+1)
            canvas[i][j].coords(cr[i][j], 1, 1, 0, 20)
            res[i][j]["text"] = ""
    svm["text"] = "SVM__"
    mnb["text"] = "MNB__"


def key_word():
    d = (" ").join(doc)
    matrix = TV.transform([d])
    word_val = []
    for i in range(len(matrix.indices)):
        word_val.append([matrix.data[i], matrix.indices[i]])
    word_val.sort(reverse=True)
    words = []
    for i in range(min(10, len(word_val))):
        words.append(feature_names[word_val[i][1]])
    text_key_word.insert(tkinter.END, (" ").join(words))


button1 = tkinter.Button(root, text="文本预处理", height=4,
                         width=20, command=pre_excu)
button1.place(x=525, y=5)

button2 = tkinter.Button(root, text="分类处理", height=4,
                         width=20, command=classify)
button2.place(x=525, y=92)

button3 = tkinter.Button(root, text="清除", height=4, width=20, command=clear)
button3.place(x=525, y=179)

Model_Name = ["TextCNN", "fastText", "HAN"]
model = [0 for i in range(3)]
for i in range(3):
    model[i] = tkinter.Label(root, text=Model_Name[i], font=("Helvetica", 20))
    model[i].place(x=40+i*200, y=280)

label = [[0 for col in range(3)]for row in range(3)]
canvas = [[0 for col in range(3)]for row in range(3)]
cr = [[0 for col in range(3)]for row in range(3)]
res = [[0 for col in range(3)]for row in range(3)]
for i in range(3):
    for j in range(3):
        label[i][j] = tkinter.Label(root, text="top_%d" % (j+1))
        label[i][j].place(x=5+i*200, y=310+j*25)
        canvas[i][j] = tkinter.Canvas(root, width=100, height=20, bg="white")
        canvas[i][j].create_rectangle(1, 1, 100, 20, outline="black", width=1)
        cr[i][j] = canvas[i][j].create_rectangle(1, 1, 0, 20, fill="blue")
        canvas[i][j].place(x=40+i*200, y=310+j*25)
        res[i][j] = tkinter.Label(root, text="")
        res[i][j].place(x=150+i*200, y=310+j*25)

svm = tkinter.Label(root, text="SVM:__", font=("Helvetica", 16))
svm.place(x=600, y=310)

mnb = tkinter.Label(root, text="MNB:__", font=("Helvetica", 16))
mnb.place(x=600, y=350)

button4 = tkinter.Button(root, text="提取关键词", command=key_word)
button4.place(x=40, y=400)

text_key_word = tkinter.Text(root, width=60, height=2)
text_key_word.place(x=140, y=400)

root.mainloop()
