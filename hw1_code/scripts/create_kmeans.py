#!/bin/python
import numpy
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys
# Generate k-means features for videos; each video is represented by a single vector

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: {0} path/to/kmeans_model cluster_num path/to/all_list".format(sys.argv[0]))
        print("kmeans_model -- kmeans model built in train_kmeans.py")
        print("cluster_num -- number of clusters")
        print("all_list -- the list of all videos (train + val + test)")
        exit(1)

    kmeans_model = sys.argv[1]
    cluster_num = int(sys.argv[2])
    file_list = sys.argv[3]

    # load the kmeans model
    kmeans = pickle.load(open(kmeans_model,"rb"))
    fread = open(file_list, "r")

    i = -1
    model_feat = {}
    df = numpy.zeros(cluster_num)
    for line in fread.readlines():
        video_id = line.replace('\n', '')
        mfcc_path = "mfcc_{}f_{}s/".format(frame, step) + video_id + ".mfcc.csv"
        i += 1
        if i != 0 and i % 500 == 0:
            print("Processed {} videos.".format(i))
        if not os.path.exists(mfcc_path):
            continue
        model_feat[video_id] = numpy.zeros(cluster_num)
        array = numpy.genfromtxt(mfcc_path, delimiter=";")
        cluster_ids = kmeans.predict(array)
        ids, counts = numpy.unique(cluster_ids, return_counts=True)
        # normalize to get tf
        model_feat[video_id][ids] = counts
        model_feat[video_id] /= sum(model_feat[video_id])
        # calculate document freq for each cluster
        df += model_feat[video_id]

    # performing tf-idf on each histogram vector of dimension k
    total_frames = numpy.sum(df)
    for id in model_feat:
        model_feat[id] *= numpy.log(total_frames / df)

    with open('feat/kmeans_dict_{}.pickle'.format(cluster_num), 'wb') as w:
        pickle.dump(model_feat, w)

    print("K-means features generated successfully! -> {}".format('feat/kmeans_dict_{}.pickle'.format(cluster_num)))
