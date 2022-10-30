import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
import logging
import optuna
import mlflow
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from models.lib.utils.metrics import get_metrics
from utils.parameters import get_data_path, get_parameters
from utils.io import get_data_state, load_data, save_model

METRICS_DIFFERENCE_THRESHOLD = 0.02

MODELS_NAMES_TO_CLASSES_MAPPING = {
    "xgboost": XGBClassifier,
    "random_forest": RandomForestClassifier,
    "logistic_regression": LogisticRegression,
}
MODELS_DEFAULT_PARAMS = {
    "xgboost": {"scale_pos_weight": 11.381618,},
    "random_forest": {
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
        "class_weight": "balanced",
    },
    "logistic_regression": {
        "fit_intercept": True,
        "solver": "saga",
        "max_iter": 500,
        "random_state": 42,
        "n_jobs": -1,
        "class_weight": "balanced",
    },
}

logger = logging.getLogger(__name__)


def main():
    logger.info("Loading parameters")
    parameters = get_parameters("train_model")
    study_path: str = parameters["study_input_path"]
    model_name: str = parameters["model_name"]
    model_difference_threshold: float = parameters[
        "model_difference_threshold"
    ]
    model_class: str = parameters["model_class"]
    mlflow.start_run(run_name=model_class)
    mlflow.log_param("model_class", model_class)
    data_path: str = get_data_path(parameters["data_path"])
    logger.info("Loading optuna study")
    study: optuna.study = optuna.load_study(
        study_name=model_name, storage=f"sqlite:///{study_path}",
    )

    logger.info("Creating best model based on study")
    model_params = MODELS_DEFAULT_PARAMS.get(model_name, {})
    model_params_optimized = get_best_model_params(
        study, model_difference_threshold=model_difference_threshold
    )
    model_params.update(model_params_optimized)
    model_class = MODELS_NAMES_TO_CLASSES_MAPPING[model_name]
    mlflow.log_param("model_params", model_params)
    model = model_class(**model_params)

    logger.info("Loading data")
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_valid: pd.DataFrame
    X_test: pd.DataFrame
    y_test: pd.DataFrame

    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data(
        path=data_path
    )
    mlflow.log_param("data_state", get_data_state())

    logger.info("Fitting model to train data")
    model.fit(X_train, y_train)

    logger.info("Evaluating model")
    train_metrics = get_metrics(model, X_train, y_train)
    logger.info(f"Metrics on train set: {train_metrics}")
    for metric, value in train_metrics.items():
        mlflow.log_metric(f"train_{metric}", value)
    logger.info(
        f"Metrics on validation set: {get_metrics(model, X_valid, y_valid)}"
    )
    test_metrics = get_metrics(model, X_test, y_test)
    logger.info(f"Metrics on test set: {test_metrics}")
    for metric, value in test_metrics.items():
        mlflow.log_metric(f"test_{metric}", value)

    logger.info("Saving model")
    save_model(
        model,
        f"resources/models/serialized/{model_name}.pkl",
        f"resources/models/parameters/{model_name}.json",
    )
    logger.info("Publishing artifacts")
    mlflow.log_artifact(f"resources/models/serialized/{model_name}.pkl")
    mlflow.log_artifact(f"resources/models/parameters/{model_name}.json")

    mlflow.end_run()


def get_best_model_params(study, model_difference_threshold):

    trials = list(
        filter(
            lambda trial: trial.values[1] < model_difference_threshold,
            study.trials,
        ),
    )
    metrics = [trial.values[0] for trial in trials]
    max_metric_index = np.argmax(metrics)
    return trials[max_metric_index].params


if __name__ == "__main__":
    main()
