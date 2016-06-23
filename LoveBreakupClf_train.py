#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation
# import pickle
# from sklearn.externals import joblib
import h5py


def main():
    # Load Dataset
    trainDict = {
        './vec/NotA0622-4221.vec': 0,
        './vec/Love0622-2100.vec': 1,
        './vec/Breakup0622-2327.vec': 2
    }
    testDict = {
        './vec/Love0623-1251.vec': 1,
        './vec/Breakup0623-1079.vec': 2
    }
    X_train, y_train = toArray(trainDict)
    X_test, y_test = toArray(testDict)
    # Build Model
    clf = Sequential()
    clf.add(Dense(output_dim=64, input_dim=300))
    clf.add(Activation("relu"))
    clf.add(Dense(output_dim=64))
    clf.add(Activation("relu"))
    clf.add(Dense(output_dim=3))
    clf.add(Activation("softmax"))
    clf.compile(loss='sparse_categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    clf.fit(X_train, y_train, batch_size=32, nb_epoch=10)
    score = clf.evaluate(X_test, y_test, batch_size=32)
    print score
    clfJson = clf.to_json()
    open('LoveBreakupClf.json', 'w').write(clfJson)
    clf.save_weights('LoveBreakupClf.h5')
    # mf = file('LoveBreakupClf.pkl', 'wb')
    # pickle.dump(clf, mf, True)
    # joblib.dump(clf, 'LoveBreakupClf.pkl')


def toArray(aDict):
    xList = []
    yList = []
    for k, v in aDict.iteritems():
        with open(k, 'r') as f:
            for l in f:
                if l.strip() != '' and l.strip() != '\n':
                    vec = []
                    for x in l.split(','):
                        vec.append(float(x))
                    xList.append(vec)
                    yList.append(v)
    return (xList, yList)

if __name__ == "__main__":
    main()
