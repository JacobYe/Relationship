#!/usr/bin/python
# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
# import h5py


def main():
    # Load Dataset
    trainDict = {
        # Single
        './EditorialVec/Single0622-615.vec': 1,
        './PMVec/Single0623-843.vec': 1,
        './PMVec/Single0725-606.vec': 1,
        # Crush
        './EditorialVec/Crush0622-391.vec': 2,
        './PMVec/Crush0623-850.vec': 2,
        './PMVec/Crush0725-602.vec': 2,
        # Love
        './EditorialVec/Love0622-2100.vec': 3,
        './PMVec/Love0623-1251.vec': 3,
        # Breakup
        './EditorialVec/Breakup0622-2327.vec': 4,
        './PMVec/Breakup0623-1079.vec': 4,
        # Marriage
        './EditorialVec/Marriage0622-715.vec': 5,
        './PMVec/Marriage0623-906.vec': 5,
        # Divorce
        './EditorialVec/Divorce0622-46.vec': 6,
        './PMVec/Divorce0623-750.vec': 6,
        './PMVec/Divorce0725-600.vec': 6
    }
    testDict = {
        './TestVec/Breakup0730-100.vec': 4,
        './TestVec/Crush0730-100.vec': 2,
        './TestVec/Divorce0730-100.vec': 6,
        './TestVec/Love0730-100.vec': 3,
        './TestVec/Marriage0730-100.vec': 5,
        './TestVec/Single0730-100.vec': 1
    }
    X_train, y_train = toArray(trainDict)
    X_test, y_test = toArray(testDict)
    # Build Model
    clf = Sequential()
    # Layer 1
    clf.add(Dense(output_dim=256, input_dim=300, init='uniform'))
    clf.add(Activation("sigmoid"))
    # Layer 2
    clf.add(Dense(output_dim=256, init='uniform'))
    clf.add(Activation("sigmoid"))
    # Layer 3
    clf.add(Dense(output_dim=256, init='uniform'))
    clf.add(Activation("sigmoid"))
    # Layer 4
    clf.add(Dense(output_dim=128, init='uniform'))
    clf.add(Activation("sigmoid"))
    clf.add(Dropout(0.5))
    # Layer 5
    clf.add(Dense(output_dim=128, init='uniform'))
    clf.add(Activation("sigmoid"))
    # Layer 6
    # clf.add(Dense(output_dim=64, init='uniform'))
    # clf.add(Activation("relu"))
    # Output Layer
    clf.add(Dense(output_dim=6, init='uniform'))
    clf.add(Activation("softmax"))
    clf.compile(loss='sparse_categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    clf.fit(X_train, y_train,
            batch_size=32,
            nb_epoch=500,
            # validation_split=0.7,
            shuffle=True)
    score = clf.evaluate(X_test, y_test, batch_size=32, verbose=1)
    print score
    clfJson = clf.to_json()
    open('RelationshipPass2.json', 'w').write(clfJson)
    clf.save_weights('RelationshipPass2.h5', overwrite=True)


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
