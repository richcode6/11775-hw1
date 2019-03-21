#!/bin/bash

#
# ex: early_fusion.sh -f places.resnet -m true -k true
# This will concatenate places and resnet features and produce MAP scores and Kaggle prediction file

# Reading of all arguments:
while getopts f:m:k: option		# p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	f) FEATURE=$OPTARG;;					# string . ex places.resnet , places 
	m) MAP=${OPTARG};;                 # boolean
	k) KAGGLE=$OPTARG;;                # boolean
	esac
	done

python scripts/concat_feats.py $FEATURE

if [ "$MAP" == true ] ; then
	echo "#######################################"
	echo "# MED with $1 Features: MAP results  #"
	echo "#######################################"


	for event in P001 P002 P003 NULL; do
	  echo "=========  Event $event  ========="
	 
	  # 1. TODO: Train SVM with OVR using only videos in training set.
	  python scripts/train_classifier.py $event features/$FEATURE.pkl models/$event.$FEATURE.model ../all_trn.lst || exit 1;
	 
	  # 2. TODO: Test SVM with val set and calculate its MAP scores for own info.
	  python scripts/test_classifier.py models/$event.$FEATURE.model features/$FEATURE.pkl prediction/${event}_$FEATURE_EF.lst ../all_val.lst || exit 1;

	  # compute the average precision by using sklearn
	  python scripts/evaluator.py list/${event}_val_label prediction/${event}_$FEATURE_EF.lst
	done
fi

if [ "$KAGGLE" == true ] ; then

	for event in P001 P002 P003 NULL; do
	  echo "=========  Event $event  ========="
	 
	  # 1. TODO: Train SVM with OVR using only videos in training set.
	  python scripts/train_classifier.py $event features/$FEATURE.pkl models/$event.$FEATURE.v.model list/trn_val.lst || exit 1;

	done

	python scripts/test_predict.py $FEATURE features/$FEATURE.pkl prediction/"$FEATURE"_EF_test.csv ../all_test_fake.lst || exit 1
fi



