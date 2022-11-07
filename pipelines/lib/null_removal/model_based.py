from typing import Dict, Any, List
import logging
import json
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pipelines.lib.constants as constants

logger = logging.getLogger(__name__)


class ModelBasedImputer(TransformerMixin):
    def __init__(
        self,
        hyperparams_cat_dict: Dict[str, Dict[str, Any]],
        hyperparams_reg_dict: Dict[str, Dict[str, Any]],
    ):
        self.model_dict: Dict[str, BaseEstimator] = {}
        self.columns_to_impute: List[str] = list(
            hyperparams_cat_dict.keys()
        ) + list(hyperparams_reg_dict.keys())

        for col, params in hyperparams_cat_dict.items():
            self.model_dict[col] = DecisionTreeClassifier(**params)
        for col, params in hyperparams_reg_dict.items():
            self.model_dict[col] = DecisionTreeRegressor(**params)

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        logger.info("Fitting one-hot encoder")
        targets_to_impute = X[self.columns_to_impute]
        X = X.drop(columns=self.columns_to_impute)
        self.columns_to_encode = list(
            set(constants.CATEGORICAL_COLUMNS_RAW)
            - set(self.columns_to_impute)
        )
        self.X_one_hot_transformer = ColumnTransformer(
            transformers=[("onehot", OneHotEncoder(), self.columns_to_encode)],
            remainder="passthrough",
        )
        X = self.X_one_hot_transformer.fit_transform(X)
        logger.info("Fitting imputing estimators")
        for column, model in self.model_dict.items():
            logger.info(f"Fitting estimator for {column}")
            column_to_impute = targets_to_impute[column]
            X_to_train = X[~column_to_impute.isna()]
            y_to_train = column_to_impute[~column_to_impute.isna()]
            model.fit(X_to_train, y_to_train)
        return self

    def transform(self, X: pd.DataFrame, y: pd.DataFrame = None, **kwargs):
        targets_to_impute = X[self.columns_to_impute]
        X_out = X.copy()
        X = X.drop(columns=self.columns_to_impute)
        X = self.X_one_hot_transformer.transform(X)
        for column, model in self.model_dict.items():
            logger.info(f"Imputing {column}")
            column_to_impute = targets_to_impute[column]
            if column_to_impute.isna().any():
                X_to_predict = X[column_to_impute.isna()]
                predictions = model.predict(X_to_predict)
                X_out.loc[column_to_impute.isna(), column] = predictions
        print(X_out.isna().sum().sum())
        if y is not None:
            return X_out, y
        return X_out

    @classmethod
    def create_from_files(
        cls, hyperparams_cat_dict_path: str, hyperparams_reg_dict_path: str
    ):
        with open(hyperparams_cat_dict_path, "r") as f:
            hyperparams_cat_dict: Dict = json.load(f)
        with open(hyperparams_reg_dict_path, "r") as f:
            hyperparams_reg_dict: Dict = json.load(f)
        return cls(
            hyperparams_cat_dict=hyperparams_cat_dict,
            hyperparams_reg_dict=hyperparams_reg_dict,
        )

