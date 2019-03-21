#!/bin/python
import numpy as np
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys


if __name__ == '__main__':
    print(sys.argv)
    if len(sys.argv) < 2:
        print("Usage: {0} feat_combination_type".format(sys.argv[0]))
        print("feat_dict_n -- dictionary of video id to feature vector stored as pickle file")
        print("output_feat_dict -- name of pickle file to store concatenated features (feat_1; feat_2;...;feat_n)")
        print("Note - Minimum 2 feature dictionaries have to be provided !!!")
        exit(1)

    feat_combo = sys.argv[1]
    output_file = "features/{}.pkl".format(feat_combo)
    input_files = list()
    for feat in feat_combo.split("."):
        input_files.append("features/{}.pkl".format(feat))

    feats = len(input_files)
    M = [pickle.load(open(input_files[i], 'rb')) for i in range(len(input_files))]
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

    with open(output_file, 'wb') as w:
        pickle.dump(X, w)

    print("Features concatenated successfully! -> {}".format(output_file))
