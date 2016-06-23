#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.neighbors import NearestCentroid
from sklearn.externals import joblib

class HobbyClassifier:
    def __init__(self):
        self.clf = joblib.load('hobby.pkl')

        self.hobbyfilter = lambda x: "喜好" if x == 0 else "非喜好"

    def predict(self, vec):
        X_test = [vec]
        pred =  self.clf.predict(X_test)
        return self.hobbyfilter(pred[0]) 

if __name__ == "__main__":
    classifier = HobbyClassifier()

    X_test = []
    with open('testlike/like.vec', 'r') as f:
        for l in f:
            if l.strip() != '' and l.strip() != '\n':
                vec = []
                for x in l.split(','):
                    vec.append(float(x))
                X_test.append(vec)
    
    for x in X_test:
        print classifier.predict(x)



