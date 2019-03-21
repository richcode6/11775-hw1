# 11775-hw2

MED Videos can be found at - http://speech-kitchen.org/sdalmia/11775_videos.tar.gz

Initial Steps: 
```
git clone https://github.com/richcode/11775-hw1.git

mkdir videos
wget http://speech-kitchen.org/sdalmia/11775_videos.tar.gz
tar -xvzf 11775_videos.tar.gz
```

run.pipeline.sh is an end-to-end configurable pipeline that uses SURF and CNN features
for Multimedia-Event Detection


Parameters
-p - Boolean, if set to true will preprocess the videos
-f - Boolean, if set to true will perform clustering and create feature representations for the video
-m - Boolean, if set to true will perform training and test mean average precision on validation
-k - Boolean, if set to true will train and generate prediction on test set for Kaggle submission.
-y - YAML file containing parameters for feature extraction
-s - Integer, denotes cluster size for feature representation
-c - Boolean, if set to true does processing for CNN features otherwise by default SURF features are
processed.
 
