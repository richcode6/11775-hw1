#!/bin/bash

# An example script for feature extraction of Homework 1

# Two additional variables
frac=$1
cluster_num=$2        # the number of clusters in k-means. Note that 50 is by no means the optimal solution.
                      # You need to explore the best config by yourself.
# You may find the number of MFCC files mfcc/*.mfcc.csv is slightly less than the number of the videos. This is because some of the videos
# don't hae the audio track. For example, HVC1221, HVC1222, HVC1261, HVC1794 

cat list/train.video list/val.video > list/train_val.video
x =  wc -l < list/train_val.video
echo $x
# In this part, we train a clustering model to cluster the MFCC vectors. In order to speed up the clustering process, we
# select a small portion of the MFCC vectors. In the following example, we only select 20% randomly from each video. 
echo "Pooling MFCCs (optional)"
python scripts/select_frames.py list/train_val.video 0.5 kmeans/select.mfcc.csv || exit 1;


# now trains a k-means model using the sklearn package
echo "Training the k-means model"
python scripts/train_kmeans.py kmeans/select.mfcc.${frac}.csv $cluster_num models/kmeans.${cluster_num}.${frac}.model || exit 1;

# Now that we have the k-means model, we can represent a whole video with the histogram of its MFCC vectors over the clusters. 
# Each video is represented by a single vector which has the same dimension as the number of clusters. 
echo "Creating k-means cluster vectors"
python scripts/create_kmeans_1.py models/kmeans.${cluster_num}.${frac}.model $cluster_num $frac list/all.video || exit 1;

# Now you can see that you get the bag-of-word representations under kmeans/. Each video is now represented
# by a {cluster_num}-dimensional vector.
exit 1

# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand. Each video is represented by
# a vector which has the same dimension as the size of the vocabulary. The elements of this vector are the number of occurrences 
# of the corresponding word. The vector is normalized to be like a probability. 
# You can of course explore other better ways, such as TF-IDF, of generating these features.
echo "Creating ASR features"
mkdir -p asrfeat
python scripts/create_asrfeat.py vocab list/all.video || exit 1;

# Great! We are done!
echo "SUCCESSFUL COMPLETION"
