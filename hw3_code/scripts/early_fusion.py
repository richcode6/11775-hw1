#!/bin/python
import numpy as np
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: {0} path/to/feat_1 path/to/feat_2... path/to/feat_n".format(sys.argv[0]))
        print("feat_dict_n -- video feature vector stored as numpy file")
        print("output_feat_dict -- name of pickle file to store concatenated features (feat_1; feat_2;...;feat_n)")
        print("Note - Minimum 2 feature dictionaries have to be provided !!!")
        exit(1)

    num_feats = len(sys.argv)-1
    feat_dir = sys.argv[1:num_feats]
    final_feat = {}

    print(feat_dir)
    for feat in feat_dir:
        for file in os.listdir(feat):
            id = file.split('.')[0]
            print(id)
            if file.endswith(".feats"):
                f = np.genfromtxt(feat+"/"+file, delimiter=';')
            elif file.endswith(".npy"):
                f = np.load(feat+"/"+file)
            else:
                continue
            if id in final_feat:
                final_feat[id] = np.append(final_feat[id], f)
            else:
                final_feat[id] = f
            print(final_feat[id].shape)

    with open(sys.argv[-1], 'wb') as w:
        pickle.dump(final_feat, w)

    print("Features concatenated successfully! -> {}".format(sys.argv[-1]))
