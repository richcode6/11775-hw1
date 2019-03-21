#!/bin/bash

#
# ex: early_fusion.sh -f places.resnet -m true -k true
# This will concatenate places and resnet features and produce MAP scores and Kaggle prediction file

# Reading of all arguments:
while getopts f:g:m:k: option		# p:f:m:k:y: is the optstring here
	do
	case "${option}"
	in
	f) FEATURE=$OPTARG;;					# string . ex places.resnet , places 
	g) GENERATE_MODEL=$OPTARG;;
	m) MAP=${OPTARG};;                 # boolean
	k) KAGGLE=$OPTARG;;                # boolean
	esac
	done

if [ "$GENERATE_MODEL" == true ]; then
	echo "Creating models using early_fusion pipeline"

	for feature_type in ${FEATURE//_/ }; do
		echo "$feature_type"
		./early_fusion.sh -f "$feature_type" -m "$MAP" -k "$KAGGLE"
	done
fi


python scripts/late_fusion.py "$FEATURE" ../all_trn.lst ../all_val.lst ../all_test_fake.lst "$MAP" "$KAGGLE"