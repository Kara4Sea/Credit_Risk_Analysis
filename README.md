# Credit Risk Analysis

## Overview of Analysis
The purpose of the analysis is to create machine learning models to evaluate the credit risk to understand which applicants are good candidates for a line of credit. During analysis, I used imbalanced-learn and scikit-learn libraries to build and evaluate resampling models. Then I also oversampled the data with RandomOverSampler and SMOTE algorithms. I also used BalancedRandomForestClassifier and EasyEnsembleClassifier to compare machine learning models in reducing bias to understand a more accurate prediction of credit risk. This provides me with the information I need to evaluate the performance of models utilized.

## Results

* Random Oversampler
  * Balanced Accuracy Score of 0.65. This reflects a fairly strong possibility of accuaracy.

    ![randomoversampler](https://user-images.githubusercontent.com/110419577/213518140-b521433b-41c5-4755-a44a-de7c46365e4b.png)


  * Precision Score - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However when we break this down further and inspect the high_risk results, they only have a precision score of 0.01, while low_risk is at 0.99. This indicates that those evaluated for low_risk are almost certaintly low risk candidate. While those evaluated at high_risk are unlikely to be high risk candidates. 
  
  * Recall Score - The recall score has a weighted average of 0.65, with consistency in results accross the high_risk (0.66) and low_risk (0.65) groups. This indicates that the test is fairly strong in determining candidates with high risk and low risk. 
  
  ![randomoversampler_imbalanced](https://user-images.githubusercontent.com/110419577/213519014-c647358d-8dd6-486b-90ea-81773e881724.png)


* SMOTE Oversampling
  * Balanced Accuracy Score of 0.65. This reflects a fairly strong possibility of accuaracy.

![SMOTE_balanced](https://user-images.githubusercontent.com/110419577/213525352-1d194bf6-7d8c-43ae-b603-b5c70db91a4a.png)


  * Precision Score - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However when we break this down further and inspect the high_risk results, they only have a precision score of 0.01, while low_risk is at 0.99. This indicates that those evaluated for low_risk are almost certaintly low risk candidate. While those evaluated at high_risk are unlikely to be high risk candidates. 
  * Recall Score - The recall score has a weighted average of 0.65, with consistency in results accross the high_risk (0.66) and low_risk (0.65) groups. This indicates that the test is fairly strong in determining candidates with high risk and low risk. 

  ![SMOTE_imbalanced](https://user-images.githubusercontent.com/110419577/213525364-bd1833fb-001b-466c-99b6-8e16c657b8b8.png)

* Cluster Centroids Resampler (Undersampling)
  * Balanced Accuracy Scores
  * Precision Scores
  * Recall Scores

* SMOTEENN (Combination Sampling)
  * Balanced Accuracy Scores
  * Precision Scores
  * Recall Scores

* Balanced Random Forest Classifier
  * Balanced Accuracy Scores
  * Precision Scores
  * Recall Scores

* Easy Ensemble AdaBoost Classifier
  * Balanced Accuracy Scores
  * Precision Scores
  * Recall Scores

## Summary
