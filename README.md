# Default prediction
The aim of this project is to create machine learning model, which predicts whether bank customer will be defaulter. Moreover model is exposed as REST API with one endpoint, which gets JSON data similar to one from link below and returns probability (calibrated) of being loan defaulter  + prediction.

## Technologies
Technologies used inside project:
* Python
* Sklearn, pandas, numpy, ...
* Optuna
* DVC
* MlFlow
* FastAPI
* Docker

## Models
Currently best model is XBboost with 0.69 ROC AUC on test set.
| Model | ROC AUC | Recall | Precision | Accuracy |
|-------|---------|--------|-----------|----------|
|XGboost|0.69|0.68|0.16|0.7|
|Logistic Regression|0.69|0.68|0.16|0.68|
|Random Forest|0.66|0.64|0.15|0.68|
|LDA|0.5|0.008|0.45|0.92|
|GaussianNB|0.51|0.98|0.08|0.12|

_Metrics on train and test sets were similar, so models aren't overfitted._

## Pipelines processing data
* **split_data** - splits data into training, validation and test ests
* **clean** - removes missing values from data using mean, median, mode and fixed values imputations (depends on column). Other available technique is model-based imputation, when missing values are predicted using decision tree.
* **transform_numerical_columns** - reduces skewness using power transform and standarizes using z-score scaling
* **balance_data** - dataset is highly imbalanced (8% of positives). This step makes it balanced. With proper configuration can be skipped. Now available options are SMOTENC [1] and oversampling.
* **transform_categorical_columns** - transforms categorical variables to numerical features. Now available options are: one-hot encoding and Category embedding [2]
* **reduce_data** - reduces data dimensionality by feature selection. Now available methods are FRUFS [3] and RFE.

## Current model specs
**Preprocessing**: 
1. model-based data imputation
2. transform_numerical_columns 
3. no_balance_data 
4. one-hot encoding 
5. RFE data reduce with decision tree as estimator

**Model**: XGBoost with balanced loss function and hyperparameters: 
```
{
    "eta": 0.73,
    "gamma": 9.83,
    "max_depth": 1,
    "lambda": 6.61,
    "n_estimators": 179,
}
```

## Link to data: 
```
https://www.kaggle.com/datasets/gauravduttakiit/loan-defaulter
```
## How to run project
In order to run project you have run
```
pip install -r requirements.txt
```
To have DVC and mlflow working you have to get from me file `secured/service_account_key.json`, which allows you to access storage with DVC and MlFlow data which are present in GCP container.

To run mlflow experiment you have to run
```
./mlflow/start_mlflow_server.sh
```
in order to start Mlflow tracking server.
## Reference
[1] https://arxiv.org/pdf/1106.1813

[2] https://arxiv.org/pdf/1604.06737.pdf

[3] https://www.deepwizai.com/projects/how-to-perform-unsupervised-feature-selection-using-supervised-algorithms

[4] https://arxiv.org/abs/2007.02837 - guided during choose of imputation method