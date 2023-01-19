# Credit Risk Analysis

## Overview of Analysis
The purpose of the analysis is to create machine learning models to evaluate the credit risk to understand which applicants are good candidates for a line of credit. During analysis, I used imbalanced-learn and scikit-learn libraries to build and evaluate resampling models. Then I also oversampled the data with RandomOverSampler and SMOTE algorithms. I also used BalancedRandomForestClassifier and EasyEnsembleClassifier to compare machine learning models in reducing bias to understand a more accurate prediction of credit risk. This provides me with the information I need to evaluate the performance of models utilized.

## Results

* Random Oversampler
  * Balanced Accuracy Score of 0.65. This reflects a fairly strong possibility of accuracy.

    ![randomoversampler](https://user-images.githubusercontent.com/110419577/213518140-b521433b-41c5-4755-a44a-de7c46365e4b.png)


  * Precision Score - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However, when we break this down further and inspect the high risk results, they only have a precision score of 0.01, while low risk is at 0.99. This indicates that those evaluated for low risk are almost certainly a low risk candidate. While those evaluated at high risk are unlikely to be high risk candidates. 
  
  * Recall Score - The recall score has a weighted average of 0.65, with consistency in results across the high risk (0.66) and low risk (0.65) groups. This indicates that the test is fairly strong in determining candidates with high risk and low risk. 
  
  ![randomoversampler_imbalanced](https://user-images.githubusercontent.com/110419577/213519014-c647358d-8dd6-486b-90ea-81773e881724.png)


* SMOTE Oversampling

  * Balanced Accuracy Score of 0.65. This reflects a fairly strong possibility of accuracy.

![SMOTE_balanced](https://user-images.githubusercontent.com/110419577/213525352-1d194bf6-7d8c-43ae-b603-b5c70db91a4a.png)


  * Precision Score - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However, when we break this down further and inspect the high risk results, they only have a precision score of 0.01, while low risk is at 0.99. This indicates that those evaluated for low risk are almost certainly low risk candidate. While those evaluated at high risk are unlikely to be high risk candidates. 
  * Recall Score - The recall score has a weighted average of 0.65, with consistency in results across the high risk (0.66) and low risk (0.65) groups. This indicates that the test is fairly strong in determining candidates with high risk and low risk. 

  ![SMOTE_imbalanced](https://user-images.githubusercontent.com/110419577/213525364-bd1833fb-001b-466c-99b6-8e16c657b8b8.png)

* Cluster Centroids Resampler (Undersampling)
  * Balanced Accuracy Score of 0.43. This reflects a fairly weak possibility of accuracy.
  
  ![cluster_balanced](https://user-images.githubusercontent.com/110419577/213534129-551eb628-5a8d-405b-86c8-8b4c5f3ff176.png)

  * Precision Scores - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However, when we break this down further and inspect the high risk results, they only have a precision score of 0.01, while low risk is at 0.99. This indicates that those evaluated for low risk are almost certainly low risk candidates. While those evaluated at high risk are unlikely to be high risk candidates. 
  * Recall Scores - The recall score has a weighted average of 0.43, with results for high risk (0.62) stronger than low risk (0.43). This indicates that the test is fairly strong in determining candidates with high risk, but fairly weak in determining candidates with low risk. 

![cluster_imbalanced](https://user-images.githubusercontent.com/110419577/213534183-fe036965-9b1e-42e6-926e-176d3c07f58c.png)

* SMOTEENN (Combination Sampling)

  * Balanced Accuracy Score of 0.61. This reflects a fairly strong possibility of accuracy.

![SMOTEENN_balanced](https://user-images.githubusercontent.com/110419577/213534626-ffe12996-02d4-4023-b100-ad85de84d3f2.png)

  * Precision Scores - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However, when we break this down further and inspect the high risk results, they only have a precision score of 0.01, while low risk is at 0.99. This indicates that those evaluated for low risk are almost certainly low risk candidates. While those evaluated at high risk are unlikely to be high risk candidates. 
  * Recall Score - The recall score has a weighted average of 0.61, with consistency in results across the high risk (0.61) and low risk (0.61) groups. This indicates that the test is fairly strong in determining candidates with high risk and low risk.

![SMOTEENN_imbalanced](https://user-images.githubusercontent.com/110419577/213534645-03a1334e-14fc-4bd4-9f28-14dde40e0f70.png)

* Balanced Random Forest Classifier
  * Balanced Accuracy Score of 0.79. This reflects a strong possibility of accuracy.

![randomforest_balanced](https://user-images.githubusercontent.com/110419577/213535272-3c0cf872-3128-46db-868a-a8364b660866.png)

  * Precision Scores - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However, when we break this down further and inspect the high risk results, they only have a precision score of 0.03, while low risk is at 0.99. This indicates that those evaluated for low risk are almost certainly low risk candidates. While those evaluated at high risk are unlikely to be high risk candidates. 
  * Recall Score - The recall score has a weighted average of 0.87. Results across the high risk (0.87) and low risk (0.70) groups are slightly inconsistent. This indicates that the test is strong in determining candidates with high risk, and strong (though slightly less so) with low risk.

![randomforest_imbalanced](https://user-images.githubusercontent.com/110419577/213535290-9b83ab9f-d20c-4921-b670-66085534f4fa.png)

* Easy Ensemble AdaBoost Classifier

  * Balanced Accuracy Score of 0.93. This reflects a very strong possibility of accuracy.

![adaboost_balanced](https://user-images.githubusercontent.com/110419577/213535347-1fa531da-a7c0-43cd-8f8b-6d71c1cece4c.png)


  * Precision Scores - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However, when we break this down further and inspect the high risk results, they have a precision score of 0.09, while low risk is at 1.00. This indicates that those evaluated for low risk are almost certainly low risk candidates. While those evaluated at high risk are somewhat unlikely to be high risk candidates. Additionally, it should be noted that although the precision score for high risk is low, it is higher than any of the other scores of models utilized.
  * Recall Score - The recall score has a weighted average of 0.94, with consistency in results across the high risk (0.92) and low risk (0.94) groups. This indicates that the test is very strong in determining candidates with high risk and low risk.

![adaboost_imbalanced](https://user-images.githubusercontent.com/110419577/213535369-5e34d6f6-5ec7-4b7b-adda-c91e604b93b2.png)


## Summary
There is a variety of results in the balanced accuracy report with Cluster Centroids at the lowest (0.43) and Easy Ensemble AdaBoost Classifier at the highest (0.93). 

Overall, the results of the machine learning models reflect a weak point in capturing a strong precision score, with the Easy Ensemble AdaBoost Classifier showing the strongest precision score at a weighted average of 0.99, and at 0.09 for high risk and 1.00 for low risk. Several precision scores are tied for weakest with a weighted average of 0.99, and at 0.01 for high risk and 0.99 for low risk. 

The Recall results showed a variety of results from weak to strong.  Cluster Centroids was the weakest with a weighted average of 0.43, and at 0.62 for high risk and 0.43 for low risk. Easy Ensemble AdaBoost Classifier is the strongest with a weighted average of 0.94, and at 0.92 for high risk and 0.94 for low risk.

I recommend the use of the Easy Ensemble AdaBoost Classifier model as it had the overall strongest results for balanced accuracy, precision score and recall score.

