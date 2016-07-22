#!/usr/bin/python
# -*- coding: utf-8 -*-

# from keras.models import Sequential
# from keras.layers import Dense, Activation
# import pickle
# from sklearn.externals import joblib
from keras.models import model_from_json


# class to use
class LoveBreakupClf:
    def __init__(self):
        clf = model_from_json(open('LoveBreakupClf.json').read())
        clf.load_weights('LoveBreakupClf.h5')
        clf.compile(loss='sparse_categorical_crossentropy',
                    optimizer='sgd')

    def predict(self, vec):
        X_test = [vec]
        prediction = self.clf.predict_classes(X_test)
        return generatePredict(prediction)

    def generatePredict(aClass):
        if aClass == 2:
            return "失恋"
        elif aClass == 1:
            return "恋爱中"
        elif aClass == 0:
            return "无关"
        else:
            return "无关"


def main():
    # Load test set

    # testList = ['./vec/Love0623-1251.vec', './vec/Breakup0623-1079.vec']
    testList = ['./vec/NotDBA0622-16.vec', './vec/NotA0622-4221.vec', './vec/NotQ0622-3051.vec']
    X_test = toArray(testList)
    y_test = []
    for i in range(len(X_test)):
        y_test.append(0)
    # Load model
    clf = model_from_json(open('LoveBreakupClf.json').read())
    clf.load_weights('LoveBreakupClf.h5')
    clf.compile(loss='sparse_categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    prediction = clf.predict_classes(X_test, batch_size=32, verbose=1)
    print prediction
    labels = generatePredict(prediction)
    print labels
    score = clf.evaluate(X_test, y_test, batch_size=32, verbose=1)
    print score


def toArray(aList):
    xList = []
    for i in aList:
        with open(i, 'r') as f:
            for l in f:
                if l.strip() != '' and l.strip() != '\n':
                    vec = []
                    for x in l.split(','):
                        vec.append(float(x))
                    xList.append(vec)
    return xList


def generatePredict(aList):
    labels = []
    for i in aList:
        if i == 0:
            labels.append("无关")
        elif i == 1:
            labels.append("恋爱中")
        elif i == 2:
            labels.append("失恋")
    return labels

if __name__ == "__main__":
    main()
