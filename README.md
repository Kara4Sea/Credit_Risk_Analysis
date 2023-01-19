# Credit Risk Analysis

## Overview of Analysis
The purpose of the analysis is to create machine learning models to evaluate the credit risk to understand which applicants are good candidates for a line of credit. During analysis, I used imbalanced-learn and scikit-learn libraries to build and evaluate resampling models. Then I also oversampled the data with RandomOverSampler and SMOTE algorithms. I also used BalancedRandomForestClassifier and EasyEnsembleClassifier to compare machine learning models in reducing bias to understand a more accurate prediction of credit risk. This provides me with the information I need to evaluate the performance of models utilized.

## Results

* Random Oversampler - instances of the minority class are randomly selected and added to the training set until the majority and minority classes are balanced
  * Balanced Accuracy Score of 0.65. This reflects a fairly strong possibility of accuaracy.

    ![randomoversampler](https://user-images.githubusercontent.com/110419577/213518140-b521433b-41c5-4755-a44a-de7c46365e4b.png)


  * Precision Score - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However when we break this down further and inspect the high_risk results, they only have a precision score of 0.01, while low_risk is at 0.99. This indicates that those evaluated for low_risk are almost certaintly low risk candidate. While those evaluated at high_risk are unlikely to be high risk candidates. 
  
  * Recall Score - The recall score has a weighted average of 0.65, with consistency in results accross the high_risk (0.66) and low_risk (0.65) groups. This indicates that the test is fairly strong in determining candidates with high risk and low risk. 
  
  ![randomoversampler_imbalanced](https://user-images.githubusercontent.com/110419577/213519014-c647358d-8dd6-486b-90ea-81773e881724.png)


* SMOTE Oversampling - In SMOTE, like random oversampling, the size of the minority is increased. The key difference between the two lies in how the minority class is increased in size. As we have seen, in random oversampling, instances from the minority class are randomly selected and added to the minority class. In SMOTE, by contrast, new instances are interpolated. That is, for an instance from the minority class, a number of its closest neighbors is chosen. Based on the values of these neighbors, new values are created.


  * Balanced Accuracy Score of 0.65. This reflects a fairly strong possibility of accuaracy.

![SMOTE_balanced](https://user-images.githubusercontent.com/110419577/213525352-1d194bf6-7d8c-43ae-b603-b5c70db91a4a.png)


  * Precision Score - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However when we break this down further and inspect the high_risk results, they only have a precision score of 0.01, while low_risk is at 0.99. This indicates that those evaluated for low_risk are almost certaintly low risk candidate. While those evaluated at high_risk are unlikely to be high risk candidates. 
  * Recall Score - The recall score has a weighted average of 0.65, with consistency in results accross the high_risk (0.66) and low_risk (0.65) groups. This indicates that the test is fairly strong in determining candidates with high risk and low risk. 

  ![SMOTE_imbalanced](https://user-images.githubusercontent.com/110419577/213525364-bd1833fb-001b-466c-99b6-8e16c657b8b8.png)

* Cluster Centroids Resampler (Undersampling) - Cluster centroid undersampling is akin to SMOTE. The algorithm identifies clusters of the majority class, then generates synthetic data points, called centroids, that are representative of the clusters. The majority class is then undersampled down to the size of the minority class.
  * Balanced Accuracy Score of 0.43. This reflects a fairly weak possibility of accuaracy.
  
  ![cluster_balanced](https://user-images.githubusercontent.com/110419577/213534129-551eb628-5a8d-405b-86c8-8b4c5f3ff176.png)

  * Precision Scores - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However when we break this down further and inspect the high_risk results, they only have a precision score of 0.01, while low_risk is at 0.99. This indicates that those evaluated for low_risk are almost certaintly low risk candidate. While those evaluated at high_risk are unlikely to be high risk candidates. 
  * Recall Scores - The recall score has a weighted average of 0.43, with results for high_risk (0.62) stronger than low_risk (0.43). This indicates that the test is fairly strong in determining candidates with high risk, but fairly weak in determining candidates with low risk. 

![cluster_imbalanced](https://user-images.githubusercontent.com/110419577/213534183-fe036965-9b1e-42e6-926e-176d3c07f58c.png)

* SMOTEENN (Combination Sampling) - Oversample the minority class with SMOTE. Clean the resulting data with an undersampling strategy. If the two nearest neighbors of a data point belong to two different classes, that data point is dropped.
  * Balanced Accuracy Score of 0.61. This reflects a fairly strong possibility of accuaracy.

![SMOTEENN_balanced](https://user-images.githubusercontent.com/110419577/213534626-ffe12996-02d4-4023-b100-ad85de84d3f2.png)

  * Precision Scores - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However when we break this down further and inspect the high_risk results, they only have a precision score of 0.01, while low_risk is at 0.99. This indicates that those evaluated for low_risk are almost certaintly low risk candidate. While those evaluated at high_risk are unlikely to be high risk candidates. 
  * Recall Score - The recall score has a weighted average of 0.61, with consistency in results accross the high_risk (0.61) and low_risk (0.61) groups. This indicates that the test is fairly strong in determining candidates with high risk and low risk.

![SMOTEENN_imbalanced](https://user-images.githubusercontent.com/110419577/213534645-03a1334e-14fc-4bd4-9f28-14dde40e0f70.png)


* Balanced Random Forest Classifier - In random undersampling, randomly selected instances from the majority class are removed until the size of the majority class is reduced, typically to that of the minority class. 
  * Balanced Accuracy Score of 0.79. This reflects a strong possibility of accuaracy.

![randomforest_balanced](https://user-images.githubusercontent.com/110419577/213535272-3c0cf872-3128-46db-868a-a8364b660866.png)

  * Precision Scores - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However when we break this down further and inspect the high_risk results, they only have a precision score of 0.03, while low_risk is at 0.99. This indicates that those evaluated for low_risk are almost certaintly low risk candidate. While those evaluated at high_risk are unlikely to be high risk candidates. 
  * Recall Score - The recall score has a weighted average of 0.87. Results accross the high_risk (0.87) and low_risk (0.70) groups are slightly inconsistent. This indicates that the test is strong in determining candidates with high risk, and strong(thought slightly less so) with low risk.

![randomforest_imbalanced](https://user-images.githubusercontent.com/110419577/213535290-9b83ab9f-d20c-4921-b670-66085534f4fa.png)

* Easy Ensemble AdaBoost Classifier - These simple trees are weak learners because they are created by randomly sampling the data and creating a decision tree for only that small portion of data. And since they are trained on a small piece of the original data, they are only slightly better than a random guess. However, many slightly better than average small decision trees can be combined to create a strong learner, which has much better decision-making power.

  * Balanced Accuracy Score of 0.93. This reflects a very strong possibility of accuaracy.

![adaboost_balanced](https://user-images.githubusercontent.com/110419577/213535347-1fa531da-a7c0-43cd-8f8b-6d71c1cece4c.png)


  * Precision Scores - The precision score has a weighted average of 0.99, which at first glance shows an almost definite possibility of true positive results. However when we break this down further and inspect the high_risk results, they only have a precision score of 0.09, while low_risk is at 0.99. This indicates that those evaluated for low_risk are almost certaintly low risk candidate. While those evaluated at high_risk are unlikely to be high risk candidates. 
  * Recall Score - The recall score has a weighted average of 0.94, with consistency in results accross the high_risk (0.92) and low_risk (0.94) groups. This indicates that the test is fairly strong in determining candidates with high risk and low risk.

![adaboost_imbalanced](https://user-images.githubusercontent.com/110419577/213535369-5e34d6f6-5ec7-4b7b-adda-c91e604b93b2.png)


## Summary
