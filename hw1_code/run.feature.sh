#!/bin/bash

# A script for feature extraction of Homework 1
# Three additional variables
cluster_num=$1        # the number of clusters in k-means. Note that 50 is by no means the optimal solution.
vocab_len=$2       # You need to explore the best config by yourself.

# You may find the number of MFCC files mfcc/*.mfcc.csv is slightly less than the number of the videos. This is because some of the videos
# don't hae the audio track. For example, HVC1221, HVC1222, HVC1261, HVC1794 

video_path="/videos"
opensmile_path="/data/VOL4/sshalini/data/speech-kitchen.org/sdalmia/opensmile-2.3.0/inst"
mpeg_path = "/data/ASR5/ramons_2/tools/ffmpeg-3.2.4/build"
export PATH=$mpeg_path/bin:$PATH
export LD_LIBRARY_PATH=$mpeg_path/libs:$LD_LIBRARY_PATH
export PATH=$opensmile_path/bin:$PATH
export LD_LIBRARY_PATH=$opensmile_path/lib:$LD_LIBRARY_PATH

mkdir -p audio mfcc kmeans  

# This part does feature extraction, it may take quite a while if you have a lot of videos. Totally 3 steps are taken:
# 1. ffmpeg extracts the audio track from each video file into a wav file
# 2. The wav file may contain 2 channels. We always extract the 1st channel using ch_wave
# 3. SMILExtract generates the MFCC features for each wav file
#    The config file MFCC12_0_D_A.conf generates 13-dim MFCCs at each frame, together with the 1st and 2nd deltas. So you 
#    will see each frame totally has 39 dims. 
#    Refer to Section 2.5 of this document http://web.stanford.edu/class/cs224s/hw/openSMILE_manual.pdf for better configuration
#    (e.g., normalization) and other feature types (e.g., PLPs )     
cat list/train | awk '{print $1}' > list/train.video
cat list/val | awk '{print $1}' > list/val.video
cat list/train.video list/val.video list/test.video > list/all.video
echo "Extracting MFCC features from videos"
start=$(date +'%s')
for line in $(cat "list/all.video"); do
        if [ -e "audio/$line.wav" ]; then
            if [ ! -e "mfcc/$line.mfcc.csv" ]; then
		        ffmpeg -y -i $video_path/${line}.mp4 -ac 1 -f wav audio/$line.wav
		        SMILExtract -C config/MFCC12_0_D_A.conf -I audio/$line.wav -O mfcc/$line.mfcc.csv
	        fi
        fi
done
echo "Extracting MFCC took $(($(date +'%s') - $start)) seconds"

# In this part, we train a clustering model to cluster the MFCC vectors. In order to speed up the clustering process, we
# select a small portion of the MFCC vectors. In the following example, we only select 20% randomly from each video. 
echo "Pooling MFCCs (optional)"
start=$(date +'%s')
python scripts/select_frames.py list/train_val.video 0.5 kmeans/select.mfcc.csv || exit 1;
echo "Pooling MFCC took $(($(date +'%s') - $start)) seconds"

# now trains a k-means model using the sklearn package
echo "Training the k-means model"
start=$(date +'%s')
python scripts/train_kmeans.py kmeans/select.mfcc.csv $cluster_num models/kmeans.${cluster_num}.model || exit 1;
echo "Training k-means took $(($(date +'%s') - $start)) seconds"

# Now that we have the k-means model, we can represent a whole video with the histogram of its MFCC vectors over the clusters. 
# Each video is represented by a single vector which has the same dimension as the number of clusters. 
echo "Creating k-means cluster vectors"
start=$(date +'%s')
python scripts/create_kmeans.py models/kmeans.${cluster_num}.model $cluster_num list/all.video || exit 1;
echo "Creating mfcc features took $(($(date +'%s') - $start)) seconds"
# Now you can see that you get the bag-of-word representations under kmeans/. The above script generates 
# a pickle file containing a dictionary of video ID to it's feature vector. Each video is now represented
# by a {cluster_num}-dimensional vector.


# Now we generate the ASR-based features. This requires a vocabulary file to available beforehand. Each video is represented by
# a vector which has the same dimension as the size of the vocabulary. The elements of this vector are the number of occurrences 
# of the corresponding word. The vector is normalized to be like a probability. 
echo "Creating ASR features"
mkdir -p asrfeat
start=$(date +'%s')
python scripts/create_asrfeat.py list/trn list/all.video $vocab_len|| exit 1;
echo "Creating ASR features took $(($(date +'%s') - $start)) seconds"
# Great! We are done!
echo "SUCCESSFUL COMPLETION"
