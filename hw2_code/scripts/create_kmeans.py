#!/bin/python
import numpy
import os
import pickle
from sklearn.cluster.k_means_ import KMeans
import sys
import threading
# Generate k-means features for videos; each video is represented by a single vector

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def create_vector(fread, output_file, t):
    i = -1
    model_feat = {}
    df = numpy.zeros(cluster_num)
    for line in fread:
        video_id = line.replace('\n', '')
        # surf_path = "cnn/" + video_id + ".surf.npz"
        surf_path = "{}/{}.{}.npz".format(t, video_id, t)
        i += 1
        if i != 0 and i % 1 == 0:
            print("Processed {} videos.".format(i))
        if not os.path.exists(surf_path):
            continue
        model_feat[video_id] = numpy.zeros(cluster_num)
        array = numpy.load(surf_path)['arr_0']
        cluster_ids = kmeans.predict(array)
        ids, counts = numpy.unique(cluster_ids, return_counts=True)
        # normalize to get tf
        model_feat[video_id][ids] = counts
        model_feat[video_id] /= sum(model_feat[video_id])
        # print(model_feat)
        # calculate document freq for each cluster
        df += model_feat[video_id]

    # performing tf-idf on each histogram vector of dimension k
    total_frames = numpy.sum(df)

    df[df == 0] = 1
    for id in model_feat:
        model_feat[id] *= numpy.log(total_frames / df)

    with open(output_file, 'wb') as w:
        pickle.dump(model_feat, w)


def chunkIt(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


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

    t = kmeans_model.split('/')[1]
    t = t.split('.')[0]
    print(t)
    output_file = 'kmeans/{}_{}.pickle'.format(t,cluster_num)

    if os.path.exists(output_file):
        print("Features already created!")
        exit(1)

    # load the kmeans model
    kmeans = pickle.load(open(kmeans_model,"rb"))
    fread = open(file_list, "r").readlines()
    lines = chunkIt(fread, 4)
    i = 0

    thread = [None for _ in range(len(lines))]

    for i in range(0, len(thread)):
        output_file = 'temp/{}_{}_{}.pickle'.format(t,cluster_num, i)
        thread[i] = threading.Thread(target=create_vector, args=(lines[i], output_file, t ))

    for i in range(0, len(thread)):
        thread[i].start()

    for i in range(0, len(thread)):
        thread[i].join()

    model_feat = {}
    for i in range(0, len(thread)):
        output_file = 'temp/{}_{}_{}.pickle'.format(t, cluster_num, i)
        x = pickle.load(open(output_file, "rb"))
        model_feat.update(x)

    output_file = 'kmeans/{}_{}.pickle'.format(t, cluster_num)
    with open(output_file, 'wb') as w:
        pickle.dump(model_feat, w)

    print("K-means features generated successfully! -> {}".format(output_file))
