#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import pickle
import sys
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# from imblearn.over_sampling import SMOTE, ADASYN

if __name__ == '__main__':
    if len(sys.argv) <= 5:
        print("Usage: {0} event_name path/to/feat_dict feat_dim output_file".format(sys.argv[0]))
        print("event_name -- name of the event (P001, P002 or P003 in Homework 1)")
        print("feat_dict -- dictionary of video id with feature stored in a pickle file")
        print("feat_dim -- dim of features")
        print("output_file -- path to save the classifier model")
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    train_list = sys.argv[5]
    train_read = open(train_list, 'r')
    lines = [line for line in train_read]
    input_dim = len(lines)
    all_vid = pickle.load(open(feat_dir, 'rb'))
    
    X = []
    Y = []
    i = 0
    for line  in lines:
        line = line.strip().split()
        # print(line)
        if line[0] in all_vid:
           X.append(all_vid[line[0]].reshape((-1,)))
           Y.append(int(line[1]==event_name))
        i += 1

    #sm = ADASYN(random_state=42)
    #X, Y =  sm.fit_resample(X, Y)
    # clf = SVC(gamma='scale',C=0.0025, probability=True)
    clf = LogisticRegression(random_state=0, solver='lbfgs', C=2.5)
    clf.fit(X, Y)
    p = clf.predict_proba(X)
    print("Training Accuracy: {}".format(clf.score(X,Y)))
    pickle.dump(clf, open(output_file, 'wb'))
    print("Model for event {} -> {}".format(event_name, output_file))
    print('SVM trained successfully for event %s!' % event_name)
