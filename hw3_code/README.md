# 11775-hw3
Create neccessary directories
```
mkdir features models prediction
```
Store extracted features in features/. The features are expected to be in dictionary format where video_id is key and numpy array is value.

To perform Early Fusion of a combination say ex. SURF and MFCC, do as below
```
./early_fusion -f surf.mfcc -m true -k true
```
Parameters:
-f, (String) feature combination type. Feature combination expected to be "." separated.

-m, (boolean) computes MAP scores.

-k, (boolean) generates Kaggle submission file



 Validation results stored as -> prediction/<event>_<feat_comb>\_EF.lst
 
 Test results stored as -> prediction/<feat_comb>\_EF.csv
 
 
 
 To perform Late Fusion of a combination say ex. SURF and MFCC, do as below
```
./late_fusion -f surf_mfcc -g true -m true -k true
```
Parameters:
-f, (String) feature combination type. Feature combination expected to be "\_" separated.

-g, (boolean) generate individual classifiers, if false will look for corresponding model in models/ folder.

-m, (boolean) computes MAP scores.

-k, (boolean) generates Kaggle submission file


 Validation results stored as -> prediction/<event>_<feat_comb>\_LF.lst
 
 Test results stored as -> prediction/<feat_comb>\_LF.csv
 
 
 
 To perform Double Fusion of a combination say ex. Late(SURF , MFCC and early(SURF+MFCC)), do as below
```
./late_fusion -f surf_mfcc_surf.mfcc -g true -m true -k true
```
Parameters:
-f, (String) feature combination type. Feature combination to be late fused expected to be "\_" separated and early fused to be '.' separated.

-g, (boolean) generate individual classifiers, if false will look for corresponding model in models/ folder.

-m, (boolean) computes MAP scores.

-k, (boolean) generates Kaggle submission file

 Validation results stored as -> prediction/<event>_<feat_comb>_LF.lst
 Test results stored as -> prediction/<feat_comb>_LF.csv
 
