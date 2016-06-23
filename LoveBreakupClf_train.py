#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation


class LoveBreakupClf:
    def __init__(self):
        trainDict = {
            './vec/NotA0622-4221.vec': 0,
            './vec/Love0622-2100.vec': 1,
            './vec/Breakup0622-2327.vec': 2
        }
        self.X_train = []
        self.y_train = []
        for k, v in trainDict.iteritems():
            with open(k, 'r') as f:
                for l in f:
                    if l.strip() != '' and l.strip() != '\n':
                        vec = []
                        for x in l.split(','):
                            vec.append(float(x))
                        self.X_train.append(vec)
                        self.y_train.append(v)
        self.model = Sequential()
        self.model.add(Dense(output_dim=64, input_dim=300))
        self.model.add(Activation("relu"))
        self.model.add(Dense(output_dim=64))
        self.model.add(Activation("relu"))
        self.model.add(Dense(output_dim=3))
        self.model.add(Activation("softmax"))
        self.model.compile(loss='categorical_crossentropy',
                           optimizer='sgd',
                           metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train, nb_epoch=5, batch_size=32)

    def predict(self, vec):
        X_test = [vec]
        classes = self.model.predict_classes(X_test, batch_size=32)
        return classes


if __name__ == "__main__":
    clf = LoveBreakupClf()
    X_test = []
    with open('./vec/Breakup0623-1070.vec', 'r') as f:
        for l in f:
            if l.strip() != '' and l.strip() != '\n':
                vec = []
                for x in l.split(','):
                    vec.append(float(x))
                X_test.append(vec)
    for x in X_test:
        print clf.predict(x)
