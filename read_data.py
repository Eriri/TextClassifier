
class Data_Set:
    def __init__(self):
        self.train_data = []
        self.train_target = []
        self.valid_data = []
        self.valid_target = []
        self.test_data = []
        self.test_target = []
        self.stopwords = set()
        raw_file = open('../data', 'r', encoding='utf-8')
        while True:
            cata_name = raw_file.readline()
            if cata_name == '':
                break
            cata_name = cata_name.strip('\n')
            for i in range(5000):
                self.train_data.append(raw_file.readline())
                self.train_target.append(cata_name)
            for i in range(500):
                self.valid_data.append(raw_file.readline())
                self.valid_target.append(cata_name)
            for i in range(500):
                self.test_data.append(raw_file.readline())
                self.test_target.append(cata_name)
        raw_file.close()
        raw_file = open('../stopwords', 'r', encoding='utf-8')
        for word in raw_file.readlines():
            self.stopwords.add(word.strip('\n'))
