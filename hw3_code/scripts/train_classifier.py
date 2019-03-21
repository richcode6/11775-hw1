#!/bin/python 

import numpy
import os
from sklearn.svm.classes import SVC
import pickle
import sys
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
# from imblearn.over_sampling import SMOTE, ADASYN

if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) < 5:
        print("Usage: {0} event_name path/to/feat_dict output_file".format(sys.argv[0]))
        print("event_name -- name of the event (P001, P002 or P003 in Homework 1)")
        print("feat_dict -- dictionary of video id with feature stored in a pickle file")
        print("output_file -- path to save the classifier model")
        exit(1)

    event_name = sys.argv[1]
    feat_dir = sys.argv[2]
    output_file = sys.argv[3]
    train_list = sys.argv[4]
    train_read = open(train_list, 'r')
    lines = [line for line in train_read]
    input_dim = len(lines)
    all_vid = pickle.load(open(feat_dir, 'rb'))
    
    X = []
    Y = []
    i = 0
    for line in lines:
        line = line.strip().split()
        # print(line)
        if line[0] in all_vid:
           # print(len(all_vid[line[0]]))
           X.append(all_vid[line[0]])
           Y.append(int(line[1]==event_name))
        i += 1


    #sm = ADASYN(random_state=42)
    #X, Y =  sm.fit_resample(X, Y)
    # clf = SVC(gamma='scale', probability=True)
    clf = SVC(probability=True,gamma='scale', C=1e3, max_iter=1000)
    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)
    # clf = SVC(gamma='auto', C=1e5, probability=True)
    # clf = SVC(gamma="scale", C=1e5, kernel='linear', probability=True)
    # clf = RandomForestClassifier(n_estimators=300)
    # clf = LogisticRegression(solver='lbfgs', C=1e3, max_iter=100, tol=1e-2, random_state=1, class_weight='balanced')
    # clf = LogisticRegression(random_state=0, solver='lbfgs', C=2.5, class_weight='balanced')
    # clf = LogisticRegression(C=10, tol=1e-2)
    # clf = GaussianNB()
    # clf = MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(512, 256), random_state=1)

    # clf = VotingClassifier(estimators=[('lr', lr), ('rf', rf), ('svm', svm)], voting = 'soft', weights=[1,1,1])

    clf.fit(X, Y)
    p = clf.predict_proba(X)
    print("Training Accuracy: {}".format(clf.score(X,Y)))
    pickle.dump(clf, open(output_file, 'wb'))
    print("Model for event {} -> {}".format(event_name, output_file))
    print('SVM trained successfully for event %s!' % event_name)
