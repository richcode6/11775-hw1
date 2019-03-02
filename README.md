# 11775-hw1

MED Videos can be found at - http://speech-kitchen.org/sdalmia/11775_videos.tar.gz

Initial Steps: 
```
git clone https://github.com/richcode/11775-hw1.git

mkdir videos
wget http://speech-kitchen.org/sdalmia/11775_videos.tar.gz
tar -xvzf 11775_videos.tar.gz
```

ASR transcripts can be found here - http://speech-kitchen.org/sdalmia/11775_asr.tar.gz
```
mkdir asr
wget http://speech-kitchen.org/sdalmia/11775_asr.tar.gz
tar -xvzf 11775_asr.tar.gz
```

Extract MFCC and ASR features. ex. run.feature.sh <cluster-size> <vocab-len>
```
 ./run.feature.sh 50 500
```

Features generated -> feat/kmeans_dict_50.pickle and feat/asr_dict_500.pickle

Train and test using these features: ex. run.med.sh <feat-type> <feat-dict> <feat-dim>
```
./run.med.sh mfcc.50 feat/kmeans_dict_50.pickle 50
```
  
 Final predictions -> prediction/MED_mfcc.50.lst.csv
 
 You can also concatenate two different feature representations. ex. python scripts/concat_feats.py <feat-dict1> <feat-dict2> <output-dict>
 ```
 python scripts.concat_feats.py feat/kmeans_dict_50.pickle feat/asr_dict_500.pickle feat/asr_500_mfcc_50.pickle
 ```
  
 
