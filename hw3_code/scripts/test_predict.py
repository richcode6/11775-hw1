#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
from sklearn.preprocessing import normalize
import pickle
import sys
import csv

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: {0} model_file feat_dir output_file".format(sys.argv[0]))
        print("model_file -- path of the trained svm file")
        print("feat_dir -- dir of feature files")
        print("output_file -- path to save the prediction score")
        exit(1)

    feat_type = sys.argv[1]
    feat_dir = sys.argv[2]
    output_file = sys.argv[3]
    test_list = sys.argv[4]
    test_read = open(test_list, 'r')
    lines = [line for line in test_read]
    input_dim = len(lines)

    models = []
    for event in ['NULL', 'P001', 'P002', 'P003']:
        clf = pickle.load(open("models/{}.{}.v.model".format(event, feat_type), 'rb'))
        models.append(clf)

    all_vid = pickle.load(open(feat_dir, 'rb'))
    i = 0
    feat_dim = len(all_vid['HVC3326'])
    X = np.zeros(shape=(input_dim, feat_dim))
    Id = []
    for line in lines:
        line = line.strip().split()
        if line[0] in all_vid:
           X[i] = all_vid[line[0]]
        Id.append(line[0])
        i += 1

    Y = [clf.predict_proba(X)[:,1] for clf in models]

    Y = np.array(Y).T
    # print(Y)
    label = list()
    for i in range(input_dim):
        # if np.max(Y[i]) < 0.55:
        #     print(np.max(Y[i]), np.argmax(Y[i]))
        #     label.append(3)
        # else:
        #
        label.append(np.argmax(Y[i]))

    Y = normalize(Y, axis=0, norm='l1')
    Y = normalize(Y, axis=1, norm='l1')
    # print(Y)
    label = list()
    for i in range(input_dim):
        # if np.max(Y[i]) < 0.55:
        #     print(np.max(Y[i]), np.argmax(Y[i]))
        #     label.append(3)
        # else:
            label.append(np.argmax(Y[i]))
    unique, counts = np.unique(label, return_counts=True)
    # print(counts)
    writer = csv.writer(open(output_file, 'w'))
    writer.writerow(['VideoID', 'Label'])
    for i in range(input_dim):
        writer.writerow([Id[i], label[i]])
    print("Prediction on test set generated! --> ", output_file)
