import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))

from typing import Any, Dict, List, Tuple
import optuna
import logging
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import pipelines.lib.constants as constants
from utils.io import load_data, save_json
from utils.parameters import get_data_path, get_pipelines_path
from utils.optuna import get_best_trial_params_cat, get_best_trial_params_reg

logger: logging.Logger = logging.getLogger(__name__)


def main():

    columns_to_drop: List[str] = constants.COLUMNS_TO_DROP
    columns_to_impute_numerical: List[str] = (
        constants.COLUMNS_TO_IMPUTE_NEG_1
        + constants.COLUMNS_TO_IMPUTE_0
        + constants.COLUMNS_TO_IMPUTE_MEDIAN
        + constants.COLUMNS_TO_FILTER_ROWS_WITH_NULLS
    )
    columns_to_impute_categorical: List[str] = (
        constants.COLUMNS_TO_IMPUTE_MODE
        + constants.COLUMNS_TO_IMPUTE_MISSING_CATEGORY
    )
    columns_to_impute: List[str] = (
        columns_to_impute_categorical + columns_to_impute_numerical
    )
    columns_to_encode: List[str] = list(
        set(constants.CATEGORICAL_COLUMNS_RAW)
        - set(columns_to_impute_categorical)
    )

    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_valid: pd.DataFrame

    X_train, y_train, X_valid, y_valid, _, _ = load_data(
        get_data_path("splitted")
    )

    logger.info("Selecting columns with missing data")
    X_train_targets_categorical: pd.DataFrame = X_train[
        columns_to_impute_categorical
    ]
    X_valid_targets_categorical: pd.DataFrame = X_valid[
        columns_to_impute_categorical
    ]
    X_train_targets_numerical: pd.DataFrame = X_train[
        columns_to_impute_numerical
    ]
    X_valid_targets_numerical: pd.DataFrame = X_valid[
        columns_to_impute_numerical
    ]

    logger.info("Dropping unwanted columns")
    X_train = X_train.drop(columns=columns_to_drop)
    X_valid = X_valid.drop(columns=columns_to_drop)
    X_train = X_train.drop(columns=columns_to_impute)
    X_valid = X_valid.drop(columns=columns_to_impute)

    logger.info("Converting non-numeric binary variables to numeric")
    X_train = X_train.replace("Y", 1).replace("N", 0)
    X_valid = X_valid.replace("Y", 1).replace("N", 0)

    logger.info("Fitting one-hot to convert categorical variables")
    one_hot_transformer: ColumnTransformer = ColumnTransformer(
        transformers=[("onehot", OneHotEncoder(), columns_to_encode)],
        remainder="passthrough",
    )
    one_hot_transformer.fit(X_train)

    logger.info("Transforming categorical variables")
    X_train: np.ndarray = one_hot_transformer.transform(X_train)
    X_valid: np.ndarray = one_hot_transformer.transform(X_valid)

    logger.info("Optimizing imputers' hyperparameters for categorical columns")
    for col in columns_to_impute_categorical:

        y_train: pd.Series = X_train_targets_categorical[col]
        y_valid: pd.Series = X_valid_targets_categorical[col]

        X_train_to_model: np.ndarray = X_train[~y_train.isna()]
        X_valid_to_model: np.ndarray = X_valid[~y_valid.isna()]
        y_train = y_train.loc[~y_train.isna()]
        y_valid = y_valid.loc[~y_valid.isna()]
        logger.info(col)
        # _optimize_params_for_imputation_model_cat(
        #     col, X_train_to_model, y_train, X_valid_to_model, y_valid
        # )

    logger.info("Optimizing imputers' hyperparameters for numerical columns")
    for col in columns_to_impute_numerical:

        y_train: pd.Series = X_train_targets_numerical[col]
        y_valid: pd.Series = X_valid_targets_numerical[col]

        X_train_to_model: np.ndarray = X_train[~y_train.isna()]
        X_valid_to_model: np.ndarray = X_valid[~y_valid.isna()]
        y_train = y_train.loc[~y_train.isna()]
        y_valid = y_valid.loc[~y_valid.isna()]
        logger.info(col)
        # _optimize_params_for_imputation_model_reg(
        #     col, X_train_to_model, y_train, X_valid_to_model, y_valid,
        # )

    logger.info("Extracting hyperparameters of best trials")

    hyperparams_cat_dict = {}
    cat_study_path = f"sqlite:///{get_pipelines_path('studies')}/imputation_models_cat.db"
    for col in columns_to_impute_categorical:
        study: optuna.Study = optuna.load_study(
            storage=cat_study_path,
            study_name=col,
        )
        best_params = get_best_trial_params_cat(study)
        best_params.update(
            {
                "class_weight": "balanced",
                "max_features": "sqrt",
                "random_state": 42,
            }
        )
        hyperparams_cat_dict[col] = best_params

    hyperparams_num_dict = {}
    num_study_path = f"sqlite:///{get_pipelines_path('studies')}/imputation_models_reg.db"
    for col in columns_to_impute_numerical:
        study: optuna.Study = optuna.load_study(
            storage=num_study_path,
            study_name=col,
        )
        best_params = get_best_trial_params_reg(study)
        best_params.update(
            {"max_features": "sqrt", "random_state": 42,}
        )
        hyperparams_num_dict[col] = best_params

    save_json(
        hyperparams_cat_dict,
        os.path.join(
            get_pipelines_path("parameters"), "hyperparams_cat_dict.json"
        ),
    )
    save_json(
        hyperparams_num_dict,
        os.path.join(
            get_pipelines_path("parameters"), "hyperparams_num_dict.json"
        ),
    )


def _optimize_params_for_imputation_model_cat(
    col: str, X_train, y_train, X_valid, y_valid
):
    study: optuna.Study = optuna.create_study(
        storage=f"sqlite:///{get_pipelines_path('studies')}/imputation_models_cat.db",
        study_name=col,
        load_if_exists=True,
        directions=["maximize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial) -> Tuple[float, float]:
        parameters: Dict[str, Any] = {
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 10, 100
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "class_weight": "balanced",
            "max_features": "sqrt",
            "random_state": 42,
        }
        model: DecisionTreeClassifier = DecisionTreeClassifier(**parameters)
        model.fit(X_train, y_train)
        y_train_pred: pd.Series = model.predict(X_train)
        y_valid_pred: pd.Series = model.predict(X_valid)
        train_f1: float = f1_score(y_train, y_train_pred, average="macro")
        valid_f1: float = f1_score(y_valid, y_valid_pred, average="macro")

        return valid_f1, train_f1 - valid_f1

    study.optimize(objective, n_jobs=-1, n_trials=300)


def _optimize_params_for_imputation_model_reg(
    col: str, X_train, y_train, X_valid, y_valid
):
    study: optuna.Study = optuna.create_study(
        storage=f"sqlite:///{get_pipelines_path('studies')}/imputation_models_reg.db",
        study_name=col,
        load_if_exists=True,
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    def objective(trial) -> Tuple[float, float]:
        parameters: Dict[str, Any] = {
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "min_samples_split": trial.suggest_int(
                "min_samples_split", 10, 100
            ),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": "sqrt",
            "random_state": 42,
        }
        model: DecisionTreeRegressor = DecisionTreeRegressor(**parameters)
        model.fit(X_train, y_train)
        y_train_pred: pd.Series = model.predict(X_train)
        y_valid_pred: pd.Series = model.predict(X_valid)
        train_mse: float = mean_squared_error(y_train, y_train_pred)
        valid_mse: float = mean_squared_error(y_valid, y_valid_pred)

        return valid_mse, (train_mse - valid_mse) ** 2

    study.optimize(objective, n_jobs=-1, n_trials=300)


if __name__ == "__main__":
    main()

