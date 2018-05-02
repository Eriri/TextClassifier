import numpy as np
import random


class Data_Set:
    def __init__(self, filename='../data'):
        self.train_data = []
        self.train_target = []
        self.test_data = []
        self.test_target = []
        self.stopwords = set()
        raw_file = open(filename, 'r', encoding='utf-8')
        while True:
            cata_name = raw_file.readline().strip('\n')
            if cata_name == '':
                break
            for i in range(5000):
                self.train_data.append(raw_file.readline())
                self.train_target.append(cata_name)
            for i in range(1000):
                self.test_data.append(raw_file.readline())
                self.test_target.append(cata_name)
        raw_file.close()
        raw_file = open('../stopwords', 'r', encoding='utf-8')
        for word in raw_file.readlines():
            self.stopwords.add(word.strip('\n'))


class Text_Vector:
    def __init__(self, filename='../tvdata.npz'):
        self.names = ['房产', '股票', '财经', '娱乐',
                      '社会', '教育', '体育', '科技', '时政', '游戏']
        self.train_data = np.load(filename)['train_data']
        self.train_target = np.load(filename)['train_target']
        self.test_data = np.load(filename)['test_data']
        self.test_target = np.load(filename)['test_target']


class Clean_Data:
    def __init__(self):
        self.data = []
        self.target = []
        raw_file = open('../datarev1', 'r', encoding='utf-8')
        while(True):
            cata_name = raw_file.readline().strip('\n')
            if cata_name == '':
                break
            for i in range(6000):
                self.data.append(raw_file.readline().strip('\n'))
                self.target.append(cata_name)
        dt = list(zip(self.data, self.target))
        random.shuffle(dt)
        self.data[:], self.target[:] = zip(*dt)
