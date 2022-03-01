# Bank-Marketing-Data-Science
## Objective
Build a classification model that predicts if the customer will subscribe to the service (term deposit).
In the model's preformance the amount of True Positive predicted is going to be prioritize, therefore, to measure the variant's preformance, the chosen metric is recall.
## Target
The target is the variable response (equal to 'y') which is rename to 'outcome'.

## Correlations
This columns are highly correlated with each other, so only one of them must remain:
    - age, age_group, eligible
    - job, salary
    - marital, education, marital-education, tag
    - pdays, pcampaign, poutcome
These correlations were proven to be codependent through variable analysis.

## Columns variations
All variations drop['duration','campaign'], columns belonging to the current campign.
df1: drops['age','eligible', 'salary', 'education', 'marital', 'tag', 'day', 'pdays','pcampaign', 'y']
df2: drops['age_group','eligible', 'salary', 'education', 'marital', 'tag', 'day', 'pdays','pcampaign', 'y']
df3: drops['age','eligible', 'salary', 'marital-education', 'tag', 'day', 'pdays','pcampaign', 'y']
df4: drops['age','eligible', 'salary', 'marital-education', 'education', 'tag', 'day', 'pdays','pcampaign', 'y']
df5: drops['age','eligible', 'job', 'education', 'marital', 'tag', 'day', 'pdays','pcampaign', 'y']
df6: drops['age','eligible', 'salary', 'contact','education', 'marital', 'tag', 'day', 'pdays','pcampaign', 'y']

| Varation  | Recall_test | pr auc     | FN   | FP   |
| --------- |:-----------:| ----------:| ----:| ----:|
| df1       | 0.654064    | 0.370697   | 549  | 3206 |
| df2       | 0.644612    | 0.370736   | 564  | 3094 |
| df3       | 0.657845    | 0.342032   | 726  | 2455 |
| df4       | 0.620038    | 0.368729   | 603  | 2779 |
| df5       | 0.661626    | 0.372399   | 537  | 3424 |
| df6       | 0.583491    | 0.360738   | 661  | 2682 |

## Feautures engineering
- Column job: it was grouped in two diffrent ways, first grouping it's values according to the yes/no ratio of each and then according to job
similarities. In both cases, the high correlation with salary remained and and the preformance with the new created column was lower than df5.
- Column day: the days were grouped into weeks but didn't offer any improvement.
- Column marital-education: they were grouped with a mixture of taking into account the similiraties in values and the releance in appeareance. The preformance of the model was worse.
- Column month: four groups were made according to the order of the month in hopes of taking relevance from May, the month with highest appearance. The preformance of the model was worse.

## Pre-proccesing
In a previously chosen dataset this technique offer better results. Is not the case for df5, the currently chosen data set.
Since the data set is unbalanced, oversampling and undersampling techniques were used in the training data in hopes of improving the model's preformance. The following techniques were used:
- SMOTE: oversampling, creates synthetic entries base on already existing entries of the minority class. Diffrent sampling ratios were run.
- Random undersampling: diffrent undersampling ratios of the majority class were tried combined with SMOTE.
- One sided selection: thoughfull undersampling of the majority class. Diffrent amount of neighbors were proven.

## Models
The completely discarded models because of its preformance are SVM and KNN. The chosen models to try diffrent hyperparameters tunning are Random Forest & XGBoost Classification.