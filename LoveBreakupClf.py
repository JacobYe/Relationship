#!/usr/bin/python
# -*- coding: utf-8 -*-

# from keras.models import Sequential
# from keras.layers import Dense, Activation
# import pickle
# from sklearn.externals import joblib
from keras.models import model_from_json


def main():
    # Load test set
    testList = {
        './vec/Love0623-1251.vec',
        './vec/Breakup0623-1079.vec'
    }
    X_test = toArray(testList)
    # Load model
    # clf = pickle.load('LoveBreakupClf.pkl')
    # clf = joblib.load("LoveBreakupClf.pkl")
    clf = model_from_json(open('LoveBreakupClf.json').read())
    clf.load_weights('LoveBreakupClf.h5')
    prediction = clf.predict_classes(X_test, batch_size=32, verbose=1)
    labels = generatePredict(prediction)
    print labels


def toArray(aList):
    xList = []
    for i in aList.iteritems():
        with open(i, 'r') as f:
            for l in f:
                if l.strip() != '' and l.strip() != '\n':
                    vec = []
                    for x in l.split(','):
                        vec.append(float(x))
                    xList.append(vec)
    return xList


def generatePredict(aList):
    for i in aList.iteritems():
        if i == 0:
            return "无关"
        elif i == 1:
            return "恋爱中"
        elif i == 2:
            return "失恋"

if __name__ == "__main__":
    main()


# class HobbyClassifier:
#     def __init__(self):
#         self.clf = joblib.load('hobby.pkl')

#         self.hobbyfilter = lambda x: "喜好" if x == 0 else "非喜好"

#     def predict(self, vec):
#         X_test = [vec]
#         pred =  self.clf.predict(X_test)
#         return self.hobbyfilter(pred[0])

# if __name__ == "__main__":
#     classifier = HobbyClassifier()

#     X_test = []
#     with open('testlike/like.vec', 'r') as f:
#         for l in f:
#             if l.strip() != '' and l.strip() != '\n':
#                 vec = []
#                 for x in l.split(','):
#                     vec.append(float(x))
#                 X_test.append(vec)
#     for x in X_test:
#         print classifier.predict(x)
