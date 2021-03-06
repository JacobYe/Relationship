#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# import h5py


def main():
    # Load Dataset
    trainDict = {
        './EditorialVec/NotA0622-4221.vec': 0,
        './EditorialVec/NotDBQ0622-3604.vec': 0,
        './EditorialVec/Single0622-615.vec': 1,
        './EditorialVec/Crush0622-391.vec': 1,
        './EditorialVec/Love0622-2100.vec': 1,
        './EditorialVec/Breakup0622-2327.vec': 1,
        './PMVec/Marriage0623-906.vec': 1,
        './PMVec/Divorce0623-750.vec': 1,
        './PMVec/Single0623-843.vec': 1,
        './PMVec/Crush0623-850.vec': 1,
        './PMVec/Love0623-1251.vec': 1,
        './PMVec/Breakup0623-1079.vec': 1,
        './EditorialVec/Marriage0622-715.vec': 1,
        './EditorialVec/Divorce0622-46.vec': 1
    }
    testDict = {
        './EditorialVec/NotDBA0622-16.vec': 0,
        './EditorialVec/NotQ0622-3051.vec': 0,
        './TestVec/Breakup0730-100.vec': 1,
        './TestVec/Crush0730-100.vec': 1,
        './TestVec/Divorce0730-100.vec': 1,
        './TestVec/Love0730-100.vec': 1,
        './TestVec/Marriage0730-100.vec': 1,
        './TestVec/Single0730-100.vec': 1
    }
    X_train, y_train = toArray(trainDict)
    X_test, y_test = toArray(testDict)
    # Build Model
    clf = Sequential()
    # Layer 1
    clf.add(Dense(output_dim=256, input_dim=300, init='uniform'))
    clf.add(Activation("relu"))
    # Layer 2
    clf.add(Dense(output_dim=256, init='uniform'))
    clf.add(Activation("relu"))
    # Layer 3
    clf.add(Dense(output_dim=256, init='uniform'))
    clf.add(Activation("relu"))
    # Layer 4
    clf.add(Dense(output_dim=128, init='uniform'))
    clf.add(Activation("relu"))
    clf.add(Dropout(0.5))
    # Layer 5
    clf.add(Dense(output_dim=128, init='uniform'))
    clf.add(Activation("relu"))
    # Layer 6
    clf.add(Dense(output_dim=64, init='uniform'))
    clf.add(Activation("relu"))
    # Output Layer
    clf.add(Dense(output_dim=2, init='uniform'))
    clf.add(Activation("softmax"))
    clf.compile(loss='sparse_categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    clf.fit(X_train, y_train, batch_size=32, nb_epoch=500)
    score = clf.evaluate(X_test, y_test, batch_size=32, verbose=1)
    print score
    clfJson = clf.to_json()
    open('RelationshipPass1.json', 'w').write(clfJson)
    clf.save_weights('RelationshipPass1.h5', overwrite=True)


def toArray(aDict):
    xList = []
    yList = []
    for k, v in aDict.iteritems():
        with open(k, 'r') as f:
            for l in f:
                if l.strip() == 'None':
                    continue
                if l.strip() != '' and l.strip() != '\n':
                    vec = []
                    for x in l.split(','):
                        vec.append(float(x))
                    xList.append(vec)
                    yList.append(v)
    return (xList, yList)

if __name__ == "__main__":
    main()
