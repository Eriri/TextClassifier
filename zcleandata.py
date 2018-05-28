import news_util
import re

stop_words_file = open('CNstopwords.txt', 'r', encoding="utf-8")
stop_words_set = set()
for word in stop_words_file.readlines():
    stop_words_set.add(word.strip('\n'))
stop_words_file.close()


def rev(text):
    words = []
    for word in text.split():
        word = re.sub('[^\u4e00-\u9fa5]', '', word)
        if len(word) > 1 and word not in stop_words_set:
            words.append(word)
    return (" ").join(words)


news = news_util.news(shuffle=False)
f = open("datarev", "w", encoding="utf-8")
for i, name in enumerate(news.names):
    f.write(name+"\n")
    for j in range(6000):
        f.write(rev(news.data[i*6000+j])+"\n")
