# Income Classifier

## Outline
- Problem Statement
- Data Exploration
- Feature Engineering
- Data Visualization
- Model Training
- Model Evaluation
- Conclusion


## Problem Statement
The objective is to create a classifier that determines if an individual's yearly income exceeds $50K


## Data Exploration
- The dataset contains information on 32,561 individuals
- It includes 14 features covering demographic details like age, gender, race, and marital status, as well as socioeconomic factors such as education, occupation, and workclass
- The target variable, "income," divides individuals into two groups: those earning over $50K per year and those earning $50K or less

### Summary statistics for numeric features
- Age ranges from 17 to 90 years, with a mean age of approximately 38.6 years
- Education-num ranges from 1 to 16
- Capital-gain and capital-loss vary widely, with minimum values of 0 and maximum values of 99999 and 4356, respectively
- Hours-per-week ranges from 1 to 99, with a mean of approximately 40.4 hours worked per week


## Feature Engineering
- Missing values were denoted by " ?" and the corresponding rows were dropped
- Absence of work class often correlates with empty occupation fields.
- The columns "education" and "education-num" represents the same data, hence “education” column was removed
- Consolidated workclass categories into government, self-employed, private, and without pay
- Grouped ages into intervals based on observed minimum and maximum values
- Classified 'hours-per-week' into minimal, standard, and extended categories


## Data Visualization
### Income Class analysis
![Income Class Distribution](https://github.com/AdibaShaikh000/machine_learning/blob/master/resources/data_visualization/distribution_of_income_class.PNG)
- There exists a noticeable contrast between the two categories.
- The majority of the dataset (75.1%) corresponds to incomes below $50K.
- The remaining 24.9% pertains to individuals earning over $50K per year.
- The skewed distribution of income classes might adversely affect model performance

### Hours per week analysis
![Hours per week box plot](https://github.com/AdibaShaikh000/machine_learning/blob/master/resources/data_visualization/box_plot_of_hours_per_week.PNG)
- The "hours-per-week" feature shows considerable variability with a standard deviation of around 12.35 hours
- The minimum value of 1 hour and maximum value of 99 hours indicates the presence of outliers

### Age analysis
![Income distribution by Age](https://github.com/AdibaShaikh000/machine_learning/blob/master/resources/data_visualization/income_distribution_by_age.PNG)
- Income tends to be lower at the starting age of the career
- Beyond the age of 30, there is a significantly increased likelihood of earning more than $50K annually
  
# Correlation 
![Correlation Matrix](https://github.com/AdibaShaikh000/machine_learning/blob/master/resources/data_visualization/Correlation_matrix.PNG)
- Income shows strong correlation with education-num, relationship, sex, capital gain, and age group
- However, correlation is weak for fnlwgt, occupation, and native-country


## Model Training
The dataset was split into 70% for training and 30% for testing.
Five models were trained to find the most effective one: 
1) Support Vector Machine (SVM)
2) K-Nearest Neighbors (KNN)
3) Logistic Regression
4) Random Forest
5) Gradient Boosting
Repeated Stratified cross-validation with 10 folds and 3 repeated was utilized during training

Comparison Result:
|         Model         |  Mean Accuracy  |
|-----------------------|-----------------|
| SVM                   | 79.65%          |
| KNN                   | 83.13%          |
| Logistic Regression   |  82.06%         |
| Random Forest         | 85.32%          |
| Gradient Boosting     | 85.43%          |

Gradient Boosting outperforms the other models


Hyperparameter tuning was conducted using GridSearchCV on the gradient boosting model to optimize its performance. 
The best parameters obtained were:
- Learning rate: 0.2
- Max depth: 4
- Min samples leaf: 1
- Min samples split: 10
- Number of estimators: 200

After hyperparameters, there was a slight increase in the accuracy


## Model Evaluation
### Classification Report
![Classification Report](https://github.com/AdibaShaikh000/machine_learning/blob/master/resources/model_evaluation/classification_report.PNG)
- Due to the imbalanced data, the most suitable evaluation metric is the macro average F1-score, which currently stands at 0.76 with the existing model
- F1-score for the ">50k" classification is 0.68 due to fewer data points in that class
  

### Confusion Matrix
![Confusion Matrix](https://github.com/AdibaShaikh000/machine_learning/blob/master/resources/model_evaluation/confusion_matrix.PNG)

Summary:
True Negatives: 6422
False Positives: 345
False Negatives: 926
True Positives: 1356

Conclusion:
The model's performance in predicting >50K income could be enhanced by reducing false negatives


### ROC Curve
![ROC Curve](https://github.com/AdibaShaikh000/machine_learning/blob/master/resources/model_evaluation/roc_curve.PNG)
- The AUC (Area Under the ROC Curve) score is 0.91
- This indicates that the model has good discriminative ability


## Conclusion
- The model demonstrates reasonable accuracy of 0.86
- Additional optimization is necessary to correctly identify individuals with higher income levels
- Addressing the imbalance in class data is crucial for enhancing model performance
- Conducting further feature engineering experiments could potentially lead to improvements in the model's predictive capabilities
