import read_data
import jieba
import progressbar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.externals import joblib


def generate_model():
    ds = read_data.Data_Set()

    mnb = Pipeline(steps=[
        ('cv', CountVectorizer()),
        ('tt', TfidfTransformer()),
        ('mnb', MultinomialNB())
    ])

    svc = Pipeline(steps=[
        ('cv', CountVectorizer()),
        ('tt', TfidfTransformer()),
        ('svc', SGDClassifier())
    ])

    pb = progressbar.ProgressBar(max_value=len(ds.train_data))
    for i in range(len(ds.train_data)):
        pb.update(i+1)
        ds.train_data[i] = (' ').join(jieba.cut(ds.train_data[i]))
    pb = progressbar.ProgressBar(max_value=len(ds.test_data))
    for i in range(len(ds.test_data)):
        pb.update(i+1)
        ds.test_data[i] = (' ').join(jieba.cut(ds.test_data[i]))
    mnb.fit(ds.train_data, ds.train_target)
    joblib.dump(mnb, 'NBmodel')
    svc.fit(ds.train_data, ds.train_target)
    joblib.dump(svc, 'SVMmodel')

    pred_traget = mnb.predict(ds.test_data)
    print('NB_predict_result')
    print(classification_report(ds.test_target, pred_traget))
    pred_traget = svc.predict(ds.test_data)
    print('SVM_predict_result')
    print(classification_report(ds.test_target, pred_traget))


if __name__ == '__main__':
    generate_model()
