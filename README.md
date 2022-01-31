# Credit Risk Analysis

## Overview

The purpose of this analysis is to use machine learning models to predict high-risk credit card loans. The performance of six machine learning models will be compared. 

The analysis was carried out in Jupyter notebook using the imbalanced-learn and scikit-learn libraries. The data used to train and test these models comes from LendingClub, a peer-to-peer lending services company. 

Data on credit risk is imbalanced - risky loans are less common than low-risk loans. Resampling methods were therefore used for the first four models: Naive Random Oversampling, SMOTE Oversampling, Undersampling, and SMOTEENN. Code for these analyses are found in credit_risk_resamplying.ipynb. The final two methods employ different ensemble classifiers: Balanced Random Forest and AdaBoost. Code for these analyses are found in credit_risk_ensemble.ipynb

## Results

### Naive Random Oversampling

Accuracy: 61.5%

![Naive_Random_Oversampling.png](https://github.com/charliuden/Credit_Risk_Analysis/blob/main/results/Naive_Random_Oversampling.png)

The model's accuracy is low. Its precision is very low for high-risk loans; the model accurately predicts low-risk loans but fails to catch high-risk loans (many false positives). The model's sensitivity (recal or 'rec') is higher than its sensitivity; low-risk loans are often predicted as high-risk. That is, there are many false negatives. Since the goal is to catch high-risk loans, this model's sensitivity is still too low and does not catch, or accurately predict enough high-risk loans. The f1 score is quite high, indicating a moderate balance between precision and accuracy. 

### SMOTE Oversampling

Accuracy: 62.4%

![SMOTE_Oversampling.png](https://github.com/charliuden/Credit_Risk_Analysis/blob/main/results/SMOTE_Oversampling.png)

The SMOTE sampling approach yields a very small increase in inaccuracy. Precision remains the same and sensitivity is slightly higher (more false-positives). The f1 score is also slightly lower. 

### Undersampling

Accuracy: 50.5%

![Undersampling.png](https://github.com/charliuden/Credit_Risk_Analysis/blob/main/results/Undersampling.png)

The undersampling approach does not perform well. The accuracy is at 50% -we may as well flip a coin. Precision remains high for low-risk loans, sensitivity decreases, and the f1 score is lower. 

### SMOTEENN

Accuracy: 64.2%

![SMOTEENN_over_under_hybrid_sampling.png](https://github.com/charliuden/Credit_Risk_Analysis/blob/main/results/SMOTEENN_over_under_hybrid_sampling.png)

The SMOTEENN method, while performing well compared to the undersapling method, shows only a small increase in performance compared with Naive Random Oversampling and SMOTE Oversampling.  

### Balanced Random Forest Classifier

Accuracy: 79.0%

![Balanced_Random_Forest_Classifier.png](https://github.com/charliuden/Credit_Risk_Analysis/blob/main/results/Balanced_Random_Forest_Classifier.png)

The Balanced Random Forest Classifier performs well according to accuracy. However, the model continues to show high precision for low-risk loans (predict many false positives or loans predicted to be low-risk when they are in fact high-risk). The f1 score is also an improvement. 


### AdaBoost Classifier

Accuracy: 93.7%

![AdaBoost_Classifier.png](https://github.com/charliuden/Credit_Risk_Analysis/blob/main/results/AdaBoost_Classifier.png)

The AdaBoost method shows the highest accuracy. Again, however, the precision is high for low-risk loans but low for high-risk loans. Sensitivity is still low - there are many false-negatives. The f1 score is the same as the Balanced Random Forest Classifier method. 

## Summary

Of the size machine learning models, the AdaBoost method shows the best accuracy at 93.7%. It also performs well precision, sensitivity, and the f1 score, though no better than the ransom forest model. While the AdaBoost model is best at predicting loan risk, I would not recommend that anyone use this model alone to predict credit risk. The model's sensitivity to high-risk, while being the highest of the six models, is still too low. That is, the model is not detecting high-risk loans frequently enough. 


