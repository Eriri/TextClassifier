from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn import metrics
import news_util

news = news_util.news(filename="datarev")


def main():
    svm = Pipeline(steps=[
        ("TV", TfidfVectorizer()),
        ("model", SGDClassifier())
    ])
    mnb = Pipeline(steps=[
        ("TV", TfidfVectorizer()),
        ("model", MultinomialNB())
    ])
    svm.fit(news.data[:51200], news.target[:51200])
    mnb.fit(news.data[:51200], news.target[:51200])

    joblib.dump(svm, "svm_model")
    joblib.dump(mnb, "mnb_model")
    TV = TfidfVectorizer()
    TV.fit(news.data[:51200])
    joblib.dump(TV, "TVmodel")


if __name__ == "__main__":
    main()
