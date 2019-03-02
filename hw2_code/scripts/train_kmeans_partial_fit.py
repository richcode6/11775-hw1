#!/bin/python 

import numpy
import os
from sklearn.cluster.k_means_ import MiniBatchKMeans
import pickle
import sys

# Performs K-means clustering and save the model to a local file

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Usage: {0} mfcc_csv_file cluster_num output_file".format(sys.argv[0]))
        print("mfcc_csv_file -- path to the mfcc csv file")
        print("cluster_num -- number of cluster")
        print("video_list -- list of videos")
        print("output_file -- path to save the k-means model")
        exit(1)

    folder_path = sys.argv[1]
    cluster_num = int(sys.argv[2])
    vid_list = open(sys.argv[3], "r").readlines()
    output_file = sys.argv[4]

    t = output_file.split('/')[1]
    t = t.split('.')[0]
    print(t)

    no_batch = False


    if os.path.exists(output_file):
        print("Kmeans already trained!")
        exit(1)

    # array = numpy.genfromtxt(mfcc_csv_file, delimiter=";")

    kmeans = MiniBatchKMeans(n_clusters=cluster_num, batch_size=70000)
    if "cnn" in t:
        no_batch = True
        if os.path.exists("cnn.batch.npz"):
            x = numpy.load("cnn.batch.npz")['arr_0']
            kmeans.fit(x)
            print(kmeans.cluster_centers_.shape)
            pickle.dump(kmeans, open(output_file, 'wb'))
            print("K-means Model -> {}".format(output_file))
            print("K-means trained successfully!")
            exit(1)

    partial = open("partial", "w")
    batch = None
    k = 0
    for file in vid_list:
        # print("{}\r".format(k))
        k += 1
        path = "{}/{}.{}.npz".format(folder_path, file.strip(), t)
        # path = folder_path+"/"+file.strip()+"..npz"
        if not os.path.exists(path):
            print(file.strip())
            partial.write(file)

        x = numpy.load(path)['arr_0']

        if not no_batch:
            sample_size = int(0.4*(x.shape[0]))

            if batch is None:
                batch = x[numpy.random.choice(x.shape[0], sample_size, replace=False), :]
            else:
                batch = numpy.vstack([batch, x[numpy.random.choice(x.shape[0], sample_size, replace=False), :]])
        else:
            sample_size = x.shape[0]
            if batch is None:
                batch = x
            else:
                batch = numpy.vstack([batch, x])

        if k%10 == 0:
            print(k, batch.shape)
            if not no_batch:
                kmeans.partial_fit(batch)
                batch = None

    if batch is not None:
        print(k, batch.shape)
        if no_batch:
            numpy.savez("{}.batch".format(t), batch)
            kmeans.fit(batch)
        else:
            kmeans.partial_fit(batch)

    # kmeans.fit(array)
    print(kmeans.cluster_centers_.shape)
    pickle.dump(kmeans, open(output_file, 'wb'))
    print("K-means Model -> {}".format(output_file))
    print("K-means trained successfully!")
