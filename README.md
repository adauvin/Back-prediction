# Back-prediction of Hemoglobin and Creatinine

## Abstract

----------------
Question: Can machine learning methods be used to accurately predict patients’ baseline haemoglobin and creatinine levels in the absence of historical blood results?

Findings: Machine learning models built using patient training data from a large intensive care dataset can accurately predict baseline haemoglobin and creatinine. 

Meaning: As electronic health records continue to become the norm there will be increasing opportunities for novel, clinically useful machine learning models that can predict baseline laboratory values and potentially other parameters when historical trends for inference are lacking. 

---------------

Importance: Patients are commonly admitted to the intensive care unit (ICU) in the absence of any baseline laboratory results. Haemoglobin and creatinine are among the most frequently deranged laboratory values in acute illness, but may also change on longer time scales, and the chronicity of the derangement markedly alters its clinical significance. The richness of data collected in the electronic health record presents an opportunity to deploy machine learning techniques to back-predict the baseline hemoglobin and creatinine, therefore bringing context to the abnormal laboratory results. 

Objective: To create a machine learning model to accurately predict baseline haemoglobin and creatinine, in the absence of historical laboratory results. 

Design: Data from the the Medical Information Mart for Intensive Care (MIMIC-III) was used.  Patients with pre-admission (baseline) haemoglobin and creatinine values were selected (6,824 and 6,508 respectively). Demographics, vital signs, and admission laboratory were used to create machine learning models that can predict pre-admission values.  Patient cohorts were split into training (75%) and testing sets (25%). Multiple preprocessing, imputation methods, ensemble methodologies, and tuning techniques were used to create prediction models. We report accuracy and feature selection of these models.   

Results:   The ensemble methodologies (notably Random Forest and XGBoost) provided the best results in terms of accuracy and AUC of the models. We were able to predict pre-admission abnormal values with high accuracy, for both hemoglobin. We predicted the exact pre-admission value of the patients with a Mean Absolute error of 1 for Hemoglobin and 0.3 for Creatinine.
We also implemented two new methods published by MIT, that increase the interpretability/transparency and feature selection of the models. Optimal Trees provided similar results for the classification task but offered much better interpretability/transparency. The  Holistic Regression highlight important features by enabling removal of collinear features, ensuring sparsity and robustness in an optimal way,  without the authors having to make make their own judgements. 

Conclusion: The emergence of ever larger intensive care datasets coupled with sophisticated machine learning techniques promises to add a new dimension of clinical information to further optimize patient care. 



## Project Repository


The 5 main folders of the git repository are:
- Cohort Selection: SQL queries, cohort selection and preliminary data analysis
- Creatinine and Hemoglobin preprocessing used to generate the X and y preprocessed for all the predictive tasks
- Final code: automated code that you can use to run the boostrapping analysis on the X and y produced by Cr and Hb code.
(- Machine learning (old folder not up-to-date): All the Python and Julia code used for data exploration, preprocessing and training predictive models)

You should first run the cohort selection to generate the master files for both the Hemoglobin and Creatinine cohorts.

Then you should run the Creatinine and Hemoglobin preprocessing codes to generate the processed X and y.

Then you should copy the X,y of interest obtained in the Hb/Cr code folder into a Data folder at the same level as the Final code folder.

(More details in the notebooks themselves)

Then you should run the final code folder which is going to run a bootstrapping on different split of the data for different models and predictions tasks. Within the same folder you can then analyze all the results
