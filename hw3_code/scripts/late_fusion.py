#!/bin/python
import numpy as np
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
import sys
import csv

class_list = ["NULL", "P001", "P002", "P003"]


def train_feat_specific_model(train_list, uses_val =False):
    ids = []
    with open(train_list, 'r') as f:
        for line in f:
            x = line.strip().split(' ')[0]
            ids.append(x)

    labels = {}
    for event in class_list:
        y = []
        with open(train_list, 'r') as f:
            for line in f:
                x = line.strip().split(' ')[1]
                y += [int(x == event)]
        labels[event] = y

    input_dim = len(ids)

    fusion_model = {}
    model_name = "models/{}.{}.v.model" if uses_val else "models/{}.{}.model"

    for event in class_list:
        feature = None
        for feat_type in feat_types:
            clf = pickle.load(open(model_name.format(event, feat_type), 'rb'))
            all_vid = pickle.load(open("features/{}.pkl".format(feat_type), 'rb'))
            feat_dim = len(all_vid['HVC3326'])
            X = np.zeros(shape=(input_dim, feat_dim))
            i = 0
            for line in ids:
                if line in all_vid:
                    X[i] = all_vid[line].reshape((-1,))
                i += 1
            print(X.shape)
            Y = clf.predict_proba(X)[:, 1]
            print(Y.shape)
            if feature is None:
                feature = [Y]
            else:
                feature += [Y]
        # print(feature.shape)
        feature = np.array(feature).T
        print(feature.shape)
        clf = LogisticRegression(solver='lbfgs', C=1e3, max_iter=100, tol=1e-2, random_state=1, class_weight='balanced')
        clf.fit(feature, labels[event])
        p = clf.predict_proba(feature)
        print("Training Accuracy: {}".format(clf.score(feature, labels[event])))
        fusion_model[event] = clf
    return fusion_model


def validate(val_list, fusion_model):
    test_read = open(val_list, 'r')
    lines = [line for line in test_read]
    test_input_dim = len(lines)
    feat_dim = len(feat_types)

    for event in class_list:
        X = np.zeros(shape=(test_input_dim, feat_dim))
        j = 0
        for feat_type in feat_types:
            clf = pickle.load(open("models/{}.{}.model".format(event, feat_type), 'rb'))
            all_vid = pickle.load(open("features/{}.pkl".format(feat_type), 'rb'))
            i = 0
            for line in lines:
                line = line.strip().split(' ')[0]
                if line in all_vid:
                    score = clf.predict_proba(all_vid[line].reshape(1, -1))[:, 1]
                else:
                    score = 0.5
                X[i, j] = score
                i += 1
            j += 1
        Y = fusion_model[event].predict_proba(X)
        s = ""
        with open("prediction/{}_{}_LF.lst".format(event, feat_abb), 'w') as f:
            for i in Y:
                s += str(i[1]) + '\n'
            f.write(s)
        os.system("python scripts/evaluator.py list/{0}_val_label prediction/{0}_{1}_LF.lst".format(event, feat_abb))


def test(test_list, fusion_model):
    test_read = open(test_list, 'r')
    lines = [line.strip().split(' ')[0] for line in test_read]
    test_input_dim = len(lines)
    feat_dim = len(feat_types)

    Y = []
    for event in class_list:
        X = np.zeros(shape=(test_input_dim, feat_dim))
        j = 0
        for feat_type in feat_types:
            clf = pickle.load(open("models/{}.{}.v.model".format(event, feat_type), 'rb'))
            all_vid = pickle.load(open("features/{}.pkl".format(feat_type), 'rb'))
            i = 0
            for line in lines:
                if line in all_vid:
                    score = clf.predict_proba(all_vid[line].reshape(1, -1))[:, 1]
                else:
                    score = 0.5
                X[i, j] = score
                i += 1
            j += 1
        Y += [fusion_model[event].predict_proba(X)[:, 1]]

    Y = np.array(Y).T
    Y = normalize(Y, axis=0, norm='l1')
    Y = normalize(Y, axis=1, norm='l1')
    # print(Y)
    label = []
    for i in range(test_input_dim):
        # if np.max(Y[i]) < 0.6:
        #     print(np.max(Y[i]), np.argmax(Y[i]))
        #     label.append(3)
        # else:
            label.append(np.argmax(Y[i]))

    # label = [(np.argmax(Y[i])) for i in range(test_input_dim)]
    writer = csv.writer(open('prediction/{}_LF_test.csv'.format(feat_abb), 'w'))
    writer.writerow(['VideoID', 'Label'])
    for i in range(test_input_dim):
        writer.writerow([lines[i], label[i]])
    print("Prediction on test set generated! --> ", 'prediction/{}_LF_test.csv'.format(feat_abb))


def train_and_validate(train_list, val_list):
    fusion_model = train_feat_specific_model(train_list)
    validate(val_list, fusion_model)


def train_and_test(train_val_list, test_list):
    fusion_model = train_feat_specific_model(train_val_list, uses_val=True)
    test(test_list, fusion_model)


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: {0} <feat_type_1> trn_list val_list test_list map kaggle".format(sys.argv[0]))
        exit(1)

    print(sys.argv)
    feat_abb = sys.argv[1]
    feat_types = feat_abb.split("_")
    train_list = sys.argv[2]
    val_list = sys.argv[3]
    test_list = sys.argv[4]
    do_map = False
    do_kaggle = False
    if len(sys.argv)>5:
        do_map = True if sys.argv[5]=="true" else False
    if len(sys.argv) >6:
        do_kaggle = True if sys.argv[6]=="true" else False
    # os.system("cat {} {} >> list/trn_val.lst".format(train_list, val_list))
    train_val_list = "list/trn_val.lst"

    print(do_kaggle)
    if do_map:
        train_and_validate(train_list, val_list)
    if do_kaggle:
        train_and_test(train_val_list, test_list)


