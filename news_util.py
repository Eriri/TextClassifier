import numpy as np
import random


class news:
    def __init__(self, filename="data", shuffle=True):
        self.names = ['房产', '股票', '财经', '娱乐',
                      '社会', '教育', '体育', '科技', '时政', '游戏']
        self.data = []
        self.target = []
        raw = open(filename, "r", encoding="utf-8")
        for name in self.names:
            raw.readline()
            for i in range(6000):
                text = raw.readline().strip("\n")
                self.data.append(text)
                self.target.append(name)
        if shuffle == True:
            dt = list(zip(self.data, self.target))
            random.shuffle(dt)
            self.data[:], self.target[:] = zip(*dt)
        print("data loaded!")


class news_c:
    def __init__(self, filename):
        self.data = np.load(filename+".npz")["data"]
        self.target = np.load(filename+".npz")["target"]
        print("data loaded!")
