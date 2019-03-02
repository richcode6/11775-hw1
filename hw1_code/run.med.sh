#!/bin/bash

# A script for multimedia event detection (MED) of Homework 1
# Before running this script, you are supposed to have the features by running run.feature.sh 

# Type of feature ex. mfcc.450, asr.600, mfcc.450.asr.600 (feat_type.vector_dim)
feat_type = $1
feat_dict = $2
feat_dim = $3

echo "#############################################"
echo "        MED with $feat_type Features          "
echo "#############################################"
mkdir -p models prediction
# iterate over the events

start1=$(date +'%s')
for event in P001 P002 P003; do
  echo "=========  Event $event  ========="
  # now train a svm model
  start=$(date +'%s')
  python scripts/train_classifier.py $event $feat_dict $feat_dim models/$event.$feat_type.model list/train || exit 1;
  echo "Training classifier took $(($(date +'%s') - $start)) seconds"
  # apply the svm model to *ALL* the testing videos;
  # output the score of each testing video to a file ${event}_pred 
  start=$(date +'%s')
  python scripts/test_classifier.py models/$event.$feat_type.model $feat_dict $feat_dim prediction/${event}_$feat_type.lst list/val || exit 1;
  echo "Testing classifier took $(($(date +'%s') - $start)) seconds"
  # compute the average precision by using sklearn
  python scripts/evaluator.py list/${event}_val_label prediction/${event}_$feat_type.lst
done
echo "Total time classifier took $(($(date +'%s') - $start1)) seconds"

start=$(date +'%s')
# now predict events for each video in the test-set 
python scripts/test_predict.py $feat_type $feat_dict $feat_dim prediction/MED_$feat_type.lst list/test || exit 1
echo "MED Prediction generated successfully! -> prediction/MED_$feat_type.lst"
echo "Predicting with classifier took $(($(date +'%s') - $start)) seconds"
