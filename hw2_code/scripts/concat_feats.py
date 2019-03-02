#!/bin/python
import numpy as np
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys


if __name__ == '__main__':

    if len(sys.argv) < 4:
        print("Usage: {0} path/to/feat_dict_1 path/to/feat_dict_2... path/to/feat_dict_n path/to/output_feat_dict".format(sys.argv[0]))
        print("feat_dict_n -- dictionary of video id to feature vector stored as pickle file")
        print("output_feat_dict -- name of pickle file to store concatenated features (feat_1; feat_2;...;feat_n)")
        print("Note - Minimum 2 feature dictionaries have to be provided !!!")
        exit(1)

    feats = len(sys.argv)-2
    M = [pickle.load(open(sys.argv[i], 'rb')) for i in range(1, feats+1)]
    dim = [len(M[i][list(M[i])[i]]) for i in range(feats)]
    total = sum(dim)
    print(total)
    keys = set().union(*M)
    X = {}
    for key in keys:
        a = [np.zeros(dim[i]) for i in range(feats)]
        for j in range(feats):
            if key in M[j]:
                a[j] = M[j][key]
        X[key] = np.concatenate(a)

    with open(sys.argv[feats+1], 'wb') as w:
        pickle.dump(X, w)

    print("Features concatenated successfully! -> {}".format(sys.argv[feats+1]))
