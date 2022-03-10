# Bank-Marketing-Data-Science

# Introduction
The increasing number of marketing campaigns over time has reduced their effect on the general public.
In addition, economic pressures and competition have led companies to invest in targeted campaigns with strict selection of contacts. This type of campaign can be improved by using Business Intelligence (BI) and Data Mining (DM) techniques.<br><br>
The data was collected from a Portuguese marketing campaign related to the subscription of bank deposits. 

Such model can increase campaign efficiency by identifying the main characteristics that affect success, helping in a better management of the available resources (e.g. human effort, phone calls, time) and selection of a high quality and affordable set of potential buying customers.

The dataset was obtained in Kaggle: https://www.kaggle.com/dhirajnirne/bank-marketing

# Objective
Build a classification model that predicts if the customer will subscribe to the service or not (term deposit).
In the model's preformance the amount of True Positive predicted is going to be prioritize, therefore, to measure the variant's preformance, the chosen metric is recall.

# Target
The target is the variable response (equal to 'y') which is rename to 'outcome'.

# Metric

Recall is a metric that quantifies the number of correct positive predictions made out of all positive predictions that could have been made.
Unlike accuracy which only comments on correct positive predictions out of all positive predictions, recall provides an indication of missing positive predictions.
In this way, recall provides some notion of the coverage of the positive class.
For unbalanced learning, recall is typically used to measure minority class coverage.

# Branches
The repository has many branches, which were named with number. These branched served for testing different preprocessing alternatives, models, hyperparameters, etc.
The main work is in the main branch and is the last and final version of it. If someone is interested in our work, a branch description is presented next:

- PAO: personal branch for research
- NICO: personal branch for research
- main: final version after revision and merging 
  - XGBoost development
  - hyperparameter tuning 
- B01:
  - Random Forest 
  - df4	
  - "used: age_group	job	marital-education	default	housing	loan	contact	month	poutcome	outcome"		
  -  pdays and poutcome cant be together, pdays should be dropped, high correaltion
- B02:
  - Random Forest
  - df6
    - ['age', 'marital', 'default', 'balance', 'housing', 'loan', 'month',  'pdays', 'poutcome', 'outcome', 'job_group']
    - recall lower than 0.6 and similar to df5
  - df5:
    - ['age', 'marital', 'default', 'balance', 'housing', 'loan', 'month',  'pdays', 'poutcome', 'outcome', 'job_type']
    - recall lower than 0.6
  - df3:
    - ['age_group', 'job', 'marital-education', 'default', 'balance',  'housing', 'loan', 'contact', 'month', 'poutcome', 'outcome']
    - no grouped features
  - df2:
    - ['age_group', 'job', 'marital-education', 'default', 'balance',  'housing', 'loan', 'contact', 'month', 'poutcome', 'outcome']
    - months grouped in three.
    - better recall result than df3
  - df1:
    - ['age_group', 'marital-education', 'default', 'balance', 'housing',  'loan', 'contact', 'month', 'poutcome', 'outcome', 'job_group']
    - jobs grouped
    - result similar to df3
- B03
  - marital-education
  - obtained bad results
- B04
  - named df5 as df0 and decided the final dataset to use
  - tried hyper paramters optimization and gridsearch. in the end arrived to one result that could not be improved
  - df0_dummy_hyper_0 smote 0.2
  - recall 0.690438 0.708255
  - {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'entropy', 'max_depth': 5, 'max_features': 'log2', 'max_leaf_nodes': 2, 'min_samples_leaf': 8, 'min_samples_split': 2, 'n_estimators': 200}
- B05
  - XGBoost
  - df0 
- B06
  - df0
  - xgboost implentation with hyperparameter tuning
  - until now the best resutls. {'colsample_bytree': 1, 'eta': 0.05, 'gamma': 5, 'max_depth': 2, 'min_child_weight': 4.0, 'scale_pos_weight': 11}
- B07
  - adding column 'campaign' to all the alternatives
  - df4 improved but, worst recall than df5 still
- B08
  - model application, use cases predictions
  - trying prediction of certain cases using the latest models

# Correlations
This columns are highly correlated with each other, so only one of them must remain:
    - age, age_group, eligible
    - job, salary
    - marital, education, marital-education, tag
    - pdays, pcampaign, poutcome
These correlations were proven to be codependent through variable analysis.


# Columns variations
All variations drop['duration','campaign'], columns belonging to the current campign.

df1: drops['age','eligible', 'salary', 'education', 'marital', 'tag', 'day', 'pdays','pcampaign', 'y']

df2: drops['age_group','eligible', 'salary', 'education', 'marital', 'tag', 'day', 'pdays','pcampaign', 'y']

df3: drops['age','eligible', 'salary', 'marital-education', 'tag', 'day', 'pdays','pcampaign', 'y']

df4: drops['age','eligible', 'salary', 'marital-education', 'education', 'tag', 'day', 'pdays','pcampaign', 'y']

df5: drops['age','eligible', 'job', 'education', 'marital', 'tag', 'day', 'pdays','pcampaign', 'y']

df6: drops['age','eligible', 'salary', 'contact','education', 'marital', 'tag', 'day', 'pdays','pcampaign', 'y']


# Feature Engineering
- Column job: it was grouped in two diffrent ways, first grouping it's values according to the yes/no ratio of each and then according to job
similarities. In both cases, the high correlation with salary remained and and the preformance with the new created column was lower than df5.
- Column day: the days were grouped into weeks but didn't offer any improvement.
- Column marital-education: they were grouped with a mixture of taking into account the similiraties in values and the releance in appeareance. The preformance of the model was worse.
- Column month: four groups were made according to the order of the month in hopes of taking relevance from May, the month with highest appearance. The preformance of the model was worse.

# Pre-proccesing
In a previously chosen dataset this technique offer better results. Is not the case for df5, the currently chosen data set.
Since the data set is unbalanced, oversampling and undersampling techniques were used in the training data in hopes of improving the model's preformance. The following techniques were used:
- SMOTE: oversampling, creates synthetic entries base on already existing entries of the minority class. Diffrent sampling ratios were run.
- Random undersampling: diffrent undersampling ratios of the majority class were tried combined with SMOTE.
- One sided selection: thoughfull undersampling of the majority class. Diffrent amount of neighbors were proven.

# Models
The completely discarded models because of its preformance are SVM and KNN. The chosen models to try diffrent hyperparameters tuning are Random Forest & XGBoost Classification.

# Results 
## Random Forest
### SMOTE (sampling_strategy=0.2)

| Varation  | Recall_test |   pr auc |   FN |   FP |
| --------- | :---------: | -------: | ---: | ---: |
| df1       |  0.654064   | 0.370697 |  549 | 3206 |
| df2       |  0.644612   | 0.370736 |  564 | 3094 |
| df3       |  0.657845   | 0.342032 |  726 | 2455 |
| df4       |  0.620038   | 0.368729 |  603 | 2779 |
| df5       |  0.661626   | 0.372399 |  537 | 3424 |
| df6       |  0.583491   | 0.360738 |  661 | 2682 |
| df4+campg |  0.642092   | 0.372000 |  568 | 2951 |

### SMOTE df0

| Sampling | recall   | precision | roc_auc  | FN    | FP     | pr_auc   |
| -------- | -------- | --------- | -------- | ----- | ------ | -------- |
| 0.2      | 0.646503 | 0.222898  | 0.673924 | 561.0 | 3577.0 | 0.357170 |
| 0.3      | 0.611216 | 0.243108  | 0.679533 | 617.0 | 3020.0 | 0.348806 |
| 0.4      | 0.590422 | 0.253106  | 0.679782 | 650.0 | 2765.0 | 0.340604 |
| 0.5      | 0.565217 | 0.269369  | 0.681039 | 690.0 | 2433.0 | 0.336515 |
| 0.6      | 0.533081 | 0.286585  | 0.678622 | 741.0 | 2106.0 | 0.328342 |
| 0.7      | 0.528040 | 0.287874  | 0.677479 | 749.0 | 2073.0 | 0.323609 |
| 0.8      | 0.522999 | 0.290922  | 0.677046 | 757.0 | 2023.0 | 0.321941 |

### SMOTE (0.2) & Random Undersampling

| Sampling | recall   | precision | roc_auc  | FN    | FP     | pr_auc   |
| -------- | -------- | --------- | -------- | ----- | ------ | -------- |
| 0.4      | 0.645243 | 0.226348  | 0.676508 | 563.0 | 3500.0 | 0.361872 |
| 0.5      | 0.641462 | 0.222806  | 0.672488 | 569.0 | 3551.0 | 0.357385 |
| 0.6      | 0.649653 | 0.218896  | 0.671241 | 556.0 | 3679.0 | 0.359535 |
| 0.7      | 0.654694 | 0.221866  | 0.675222 | 548.0 | 3644.0 | 0.357908 |
| 0.8      | 0.642092 | 0.222781  | 0.672637 | 568.0 | 3555.0 | 0.35747  |

### SMOTE (0.2) & Undersampling

| Neighbors | recall   | precision | roc_auc  | FN    | FP     | pr_auc   |
| --------- | -------- | --------- | -------- | ----- | ------ | -------- |
| 1.0       | 0.638941 | 0.224187  | 0.672982 | 573.0 | 3509.0 | 0.359020 |
| 2.0       | 0.630120 | 0.236072  | 0.679968 | 587.0 | 3236.0 | 0.359690 |
| 3.0       | 0.641462 | 0.225072  | 0.674409 | 569.0 | 3505.0 | 0.360370 |
| 4.0       | 0.637681 | 0.223745  | 0.672268 | 575.0 | 3511.0 | 0.357296 |
| 5.0       | 0.644612 | 0.221765  | 0.672436 | 564.0 | 3590.0 | 0.359035 |
| 6.0       | 0.640202 | 0.223493  | 0.672735 | 571.0 | 3530.0 | 0.359864 |

### Undersampling

| Neighbors | recall   | precision | roc_auc  | FN    | FP     | pr_auc   |
| --------- | -------- | --------- | -------- | ----- | ------ | -------- |
| 1.0       | 0.637051 | 0.238669  | 0.683892 | 576.0 | 3225.0 | 0.373162 |
| 2.0       | 0.638941 | 0.238084  | 0.684003 | 573.0 | 3245.0 | 0.373819 |
| 3.0       | 0.637051 | 0.242679  | 0.686815 | 576.0 | 3155.0 | 0.373061 |
| 4.0       | 0.634531 | 0.246572  | 0.688811 | 580.0 | 3077.0 | 0.371942 |
| 5.0       | 0.637051 | 0.244321  | 0.687984 | 576.0 | 3127.0 | 0.372587 |
| 6.0       | 0.637051 | 0.242621  | 0.686773 | 576.0 | 3156.0 | 0.373834 |

### SMOTE & GridSearch

| name | train_recall | test_recall | precision | roc_auc  | pr_auc   | FN  | FP   | Hyperparameters |
| ---- | ------------ | ----------- | --------- | -------- | -------- | --- | ---- | --------------- |
| df0  | 0.681253     | 0.692502    | 0.201577  | 0.664527 | 0.293591 | 488 | 4353 | check notebook  |
| df0  | 0.714209     | 0.717076    | 0.191486  | 0.657945 | 0.296933 | 449 | 4805 | check notebook  |

## XGBoost
### Gridsearch

| name         | train_recall | test_recall | precision | roc_auc  | pr_auc   | FN  | FP   | Hyperparameters |
| ------------ | ------------ | ----------- | --------- | -------- | -------- | --- | ---- | --------------- |
| df0_xgb_1    | 0.835494     | 0.838689    | 0.171653  | 0.651206 | 0.397953 | 256 | 6423 | check notebook  |
| df0_xgb_2    | 0.791194     | 0.783869    | 0.195998  | 0.678901 | 0.407330 | 343 | 5103 | check notebook  |
| df0_duration | 0.904646     | 0.897921    | 0.340991  | 0.833990 | 0.551135 | 162 | 2754 | check notebook  |
**df0_duration corresponds to benchmarking**

More XGBoost results in the previosly mentioned branches.

## Model Application
To try the chosen model real predictions were made using examples. Note that the threshold is set by default and in a real case application this should be set according to the business objective.


  
