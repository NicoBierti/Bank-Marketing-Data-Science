# Bank-Marketing-Data-Science
## Objective
Build a classification model that predicts if the customer will subscribe to the service (term deposit).
In the model's preformance the amount of True Positive predicted is going to be prioritize, therefore, to measure the variant's preformance, the chosen metric is recall.
## Target
The target is the variable response (equal to 'y') which is rename to 'outcome'.
##Variations
All variations drop['duration','campaign']
df1: drops['age_group','salary','marital-education','tag','contact', 'day','pcampaign', 'y', 'week', 'job_group']
df2: drops['age_group','eligible', 'salary','marital','education', 'tag', 'contact', 'day','pcampaign', 'y', 'week', 'job','job_type']
df3: drops['age_group','eligible', 'salary','marital','education', 'tag', 'contact', 'day','pcampaign', 'y', 'week', 'job','job_group']
df4: drops['age_group','eligible', 'marital','education', 'marital-education', 'tag', 'contact', 'day','pcampaign', 'y', 'week','job','job_group','job_type']

| Varation  | Recall      | pr auc     | FN   | FP   |
| --------- |:-----------:| ----------:| ----:| ----:|
| df1       | 0.540643    | 0.328766   | 729  | 2561 |
| df2       | 0.535602    | 0.343343   | 737  | 2413 |
| df3       | 0.542533    | 0.342032   | 726  | 2455 |
| df4       | 0.514808    | 0.343066   | 770  | 2014 |