from csv import DictReader

from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report



def read_data(name):
    text, targets = [], []

    with open('./../data/{}.csv'.format(name)) as f:
        for item in DictReader(f):
            text.append(item['text'].decode('utf8'))
            targets.append(item['category'])

    return text, targets


def main():
    text_train, targets_train = read_data('train')
    text_test, targets_test = read_data('test')
    
    print "LENGTH OF TEXT_TRAIN %d" % len(text_train)
    print "LENGTH OF TEST %d" % len(text_test)

    model = make_pipeline(
        TfidfVectorizer(),
        LogisticRegression(),
    ).fit(text_train, targets_train)

    prediction = model.predict(text_test)
    print '\n'

    print 'macro f1:', f1_score(targets_test, prediction, average='macro')
    print '\n\n'
    print 'confusion matrix...\n'
    print (classification_report(targets_test, prediction,digits=5))


if __name__ == "__main__":
    main()
