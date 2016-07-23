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
class RelationshipPass1:

    # model is load successfully
    isLoad = False

    def __init__(self, modelRoot):
        modelFile = modelRoot + '/RelationshipPass1.json'
        logging.info("model file " + modelFile)
        weightFile = modelRoot + '/RelationshipPass1.h5'
        logging.info("weight file " + weightFile)
        if modelRoot is None or not os.path.exists(modelFile) or not os.path.exists(weightFile):
            return
        # load models
        self.clf = model_from_json(open(modelFile).read())
        self.clf.load_weights(weightFile)
        self.clf.compile(loss='sparse_categorical_crossentropy',
                         optimizer='sgd')
        self.isLoad = True

    def generateSinglePredict(self, aClass, aProba):
        if aClass == 0:
            return "无关"
        elif aClass == 1 and aProba > 0.7:
            return "有关"
        else:
            return "无关"

    def predict(self, vec):
        if not self.isLoad:
            return "加载模型失败"

        X_test = numpy.asarray([vec])
        predictions = self.clf.predict_classes(X_test, batch_size=32, verbose=1)
        proba = self.clf.predict_proba(X_test, batch_size=32, verbose=1)
        logging.info(predictions)
        prediction = predictions[0]
        return self.generateSinglePredict(prediction, proba)


def main():
    pass

if __name__ == "__main__":
    main()
