#!/bin/python 

import numpy as np
import os
from sklearn.svm.classes import SVC
import pickle
import sys

# Apply the SVM model to the testing videos; Output the score for each video

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage: {0} model_file feat_dir feat_dim output_file".format(sys.argv[0]))
        print("model_file -- path of the trained svm file")
        print("feat_dir -- dir of feature files")
        print("feat_dim -- dim of features; provided just for debugging")
        print("output_file -- path to save the prediction score")
        exit(1)

    model_file = sys.argv[1]
    feat_dir = sys.argv[2]
    feat_dim = int(sys.argv[3])
    output_file = sys.argv[4]
    test_list = sys.argv[5]
    test_read = open(test_list, 'r')
    lines = [line for line in test_read]
    input_dim = len(lines)
    clf = pickle.load(open(model_file, 'rb'))
    all_vid = pickle.load(open(feat_dir, 'rb'))
    i = 0
    X = np.zeros(shape=(input_dim, feat_dim))
    for line in lines:
        line = line.strip().split()
        if line[0] in all_vid:
           X[i] = all_vid[line[0]].reshape((-1,))
        i += 1

    Y = clf.decision_function(X)
    s = ""
    with open(output_file, 'w') as f:
        for i in Y:
            s += str(i)+'\n'
        f.write(s)
    print("Tested successfully!")
