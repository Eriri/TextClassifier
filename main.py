import read_data
import jieba
from gensim import models

ds = read_data.Data_Set()
print(set(ds.train_target))
