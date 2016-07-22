#!/usr/bin/python
# -*- coding: utf-8 -*-

# from keras.models import Sequential
# from keras.layers import Dense, Activation
# import pickle
# from sklearn.externals import joblib
import os
import logging
import numpy
from keras.models import model_from_json


# class to use
class RelationshipClf:

    # model is load successfully
    isLoad = False

    def __init__(self, modelRoot):
        modelFile = modelRoot + '/RelationshipClf.json'
        logging.info("model file " + modelFile)
        weightFile = modelRoot + '/Relationship.h5'
        logging.info("weight file " + weightFile)
        if modelRoot is None or not os.path.exists(modelFile) or not os.path.exists(weightFile):
            return
        # load models
        self.clf = model_from_json(open(modelFile).read())
        self.clf.load_weights(weightFile)
        self.clf.compile(loss='sparse_categorical_crossentropy',
                         optimizer='sgd')
        self.isLoad = True

    def generateSinglePredict(self, aClass):
        if aClass == 6:
            return "离婚"
        elif aClass == 5:
            return "结婚"
        elif aClass == 4:
            return "失恋"
        elif aClass == 3:
            return "恋爱"
        elif aClass == 2:
            return "暗恋"
        elif aClass == 1:
            return "单身"
        elif aClass == 0:
            return "无关"
        else:
            return "无关"

    def predict(self, vec):
        if not self.isLoad:
            return "加载模型失败"

        X_test = numpy.asarray([vec])
        predictions = self.clf.predict_classes(X_test, batch_size=32, verbose=1)
        logging.info(predictions)
        prediction = predictions[0]
        return self.generateSinglePredict(prediction)


def main():
    # Load test set

    testList = ['./vec/Love0623-1251.vec', './vec/Breakup0623-1079.vec']
    X_test = toArray(testList)
    # Load model
    clf = model_from_json(open('LoveBreakupClf.json').read())
    clf.load_weights('LoveBreakupClf.h5')
    clf.compile(loss='sparse_categorical_crossentropy',
                optimizer='sgd')
    prediction = clf.predict_classes(X_test, batch_size=32, verbose=1)
    print prediction
    labels = generatePredict(prediction)
    print labels


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
            labels.append("单身")
        elif i == 2:
            labels.append("暗恋")
        elif i == 3:
            labels.append("恋爱")
        elif i == 4:
            labels.append("失恋")
        elif i == 5:
            labels.append("结婚")
        elif i == 6:
            labels.append("离婚")
    return labels

if __name__ == "__main__":
    main()
