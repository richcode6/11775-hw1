#!/bin/bash

# This script performs a complete Media Event Detection pipeline (MED) using video features:
# a) preprocessing of videos, b) feature representation,
# c) computation of MAP scores, d) computation of class labels for kaggle submission.

# You can pass arguments to this bash script defining which one of the steps you want to perform.
# This helps you to avoid rewriting the bash script whenever there are
# intermediate steps that you don't want to repeat.

# execute: bash run.pipeline.sh -p true -f true -m true -k true -y filepath



# Reading of all arguments:
while getopts p:f:m:k:y:s:c: option		# p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	p) PREPROCESSING=${OPTARG};;       # boolean true or false
	f) FEATURE_REPRESENTATION=${OPTARG};;  # boolean
	m) MAP=${OPTARG};;                 # boolean
	k) KAGGLE=$OPTARG;;                # boolean
    y) YAML=$OPTARG;;                  # path to yaml file containing parameters for feature extraction
	s) CLUSTER_SIZE=$OPTARG;;
	c) cnn=$OPTARG;;
	esac
	done

# export PATH="/data/ASR5/ramons_2/tools/ffmpeg-3.2.4/build/bin":$PATH
source /data/ASR5/ramons_2/sinbad_projects/myenvs/mini_conda_nmtpy_p36/bin/activate
source activate "/home/rrnigam/.conda/envs/scikit/"

if [ "$PREPROCESSING" = true ] ; then

    echo "#####################################"
    echo "#         PREPROCESSING             #"
    echo "#####################################"

    # steps only needed once
    video_path=/data/VOL4/sshalini/data/speech-kitchen.org/sdalmia/11775_videos/video  # path to the directory containing all the videos.
    mkdir -p list downsampled_videos surf cnn kmeans  # create folders to save features
    awk '{print $1}' ../all_trn.lst > list/train.video  # save only video names in one file (keeping first column)
    awk '{print $1}' ../all_val.lst > list/val.video
	awk '{print $1}' ../all_test_fake.lst > list/test.video
    cat list/train.video list/val.video list/test.video > list/all.video    #save all video names in one file
	cat list/train.video list/val.video > list/trn_val.video    #save all train+val names in one file
    downsampling_frame_len=60
    downsampling_frame_rate=15

    # 1. Downsample videos into shorter clips with lower frame rates.
    # TODO: Make this more efficient through multi-threading f.ex.
    start=`date +%s`
    for line in $(cat "list/all.video"); do
		if [ ! -e downsampled_videos/$line.ds.mp4 ]; then
			ffmpeg -y -ss 0 -i $video_path/${line}.mp4 -strict experimental -t $downsampling_frame_len -r $downsampling_frame_rate downsampled_videos/$line.ds.mp4
		fi
    done
    end=`date +%s`
    runtime=$((end-start))
    echo "Downsampling took: $runtime" #28417 sec around 8h without parallelization
	if [ "$cnn" = true ] ; then
		# 3. TODO: Extract CNN features from keyframes of downsampled videos
		python scripts/cnn_feat_extraction.py list/all.video config.yaml
	
	else
		# 2. TODO: Extract SURF features over keyframes of downsampled videos (0th, 5th, 10th frame, ...)
		python scripts/surf_feat_extraction.py list/all.video config.yaml
		
	fi
   

fi

if [ "$FEATURE_REPRESENTATION" = true ] ; then
	
	if [ "$cnn" = true ] ; then
		echo "#####################################"
		echo "#   CNN FEATURE REPRESENTATION      #"
		echo "#####################################"

		# 1. TODO: Train kmeans to obtain clusters for CNN features
		python scripts/train_kmeans_partial_fit.py cnn $CLUSTER_SIZE list/all.video kmeans/cnn."$CLUSTER_SIZE".model

		# 2. TODO: Create kmeans representation for CNN features
		python scripts/create_kmeans.py kmeans/cnn."$CLUSTER_SIZE".model $CLUSTER_SIZE list/all.video
		
	else
		echo "#####################################"
		echo "#  SURF FEATURE REPRESENTATION      #"
		echo "#####################################"

		# 1. TODO: Train kmeans to obtain clusters for SURF features
		python scripts/train_kmeans_partial_fit.py surf $CLUSTER_SIZE list/trn_val.video kmeans/surf."$CLUSTER_SIZE".model
		
		# 2. TODO: Create kmeans representation for SURF features
		python scripts/create_kmeans.py kmeans/surf."$CLUSTER_SIZE".model $CLUSTER_SIZE list/all.video
	fi	
fi

if [ "$MAP" = true ] ; then
	if [ "$cnn" = true ] ; then
		echo "#######################################"
		echo "# MED with CNN Features: MAP results  #"
		echo "#######################################"


		for event in P001 P002 P003; do
		  echo "=========  Event $event  ========="
		 
		  # 1. TODO: Train SVM with OVR using only videos in training set.
		  python scripts/train_classifier.py $event kmeans/cnn_"$CLUSTER_SIZE".pickle "$CLUSTER_SIZE" models/$event.cnn."$CLUSTER_SIZE".model ../all_trn.lst || exit 1;
		 
		  # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
		  python scripts/test_classifier.py models/$event.cnn."$CLUSTER_SIZE".model kmeans/cnn_"$CLUSTER_SIZE".pickle "$CLUSTER_SIZE" prediction/${event}_cnn_"$CLUSTER_SIZE" ../all_val.lst || exit 1;

		  # compute the average precision by using sklearn
		  python scripts/evaluator.py list/${event}_val_label prediction/${event}_cnn_"$CLUSTER_SIZE"
		  
		done
	
	else
		echo "#######################################"
		echo "# MED with SURF Features: MAP results #"
		echo "#######################################"

		for event in P001 P002 P003; do
		  echo "=========  Event $event  ========="
		 
		  # 1. TODO: Train SVM with OVR using only videos in training set.
		  python scripts/train_classifier.py $event kmeans/surf_"$CLUSTER_SIZE".pickle "$CLUSTER_SIZE" models/$event.surf."$CLUSTER_SIZE".model ../all_trn.lst || exit 1;
		 
		  # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
		  python scripts/test_classifier.py models/$event.surf."$CLUSTER_SIZE".model kmeans/surf_"$CLUSTER_SIZE".pickle "$CLUSTER_SIZE" prediction/${event}_surf_"$CLUSTER_SIZE".lst ../all_val.lst || exit 1;

		  # compute the average precision by using sklearn
		  python scripts/evaluator.py list/${event}_val_label prediction/${event}_surf_"$CLUSTER_SIZE".lst
		  
		done
	fi
fi


if [ "$KAGGLE" = true ] ; then
	
	if [ "$cnn" = true ] ; then
	echo "##########################################"
    echo "# MED with CNN Features: KAGGLE results  #"
    echo "##########################################"

    for event in P001 P002 P003; do
		# 3. TODO: Train SVM with OVR using videos in training and validation set.
		python scripts/train_classifier.py $event kmeans/cnn_"$CLUSTER_SIZE".pickle "$CLUSTER_SIZE" models/$event.cnn."$CLUSTER_SIZE".v.model ../all_trn_val.lst || exit 1;
	done
	
	# 4. TODO: Test SVM with test set saving scores for submission
	python scripts/test_predict.py cnn kmeans/cnn_"$CLUSTER_SIZE".pickle "$CLUSTER_SIZE" prediction/MED_cnn_"$CLUSTER_SIZE".lst ../all_test_fake.lst || exit 1
   
	else
		echo "##########################################"
		echo "# MED with SURF Features: KAGGLE results #"
		echo "##########################################"
		for event in P001 P002 P003; do
			# 3. TODO: Train SVM with OVR using videos in training and validation set.
			python scripts/train_classifier.py $event kmeans/surf_"$CLUSTER_SIZE".pickle "$CLUSTER_SIZE" models/$event.surf."$CLUSTER_SIZE".v.model ../all_trn.lst || exit 1;
		done
		
		# 4. TODO: Test SVM with test set saving scores for submission
		python scripts/test_predict.py surf kmeans/surf_"$CLUSTER_SIZE".pickle "$CLUSTER_SIZE" prediction/MED_surf_"$CLUSTER_SIZE".lst ../all_test_fake.lst || exit 1

	fi  
fi
