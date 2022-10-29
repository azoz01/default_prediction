import sys
import os

sys.path.append(os.path.abspath(os.getcwd()))
import logging
import optuna
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from models.utils.metrics import get_metrics
from utils.parameters import get_data_path, get_parameters
from utils.io import load_data, save_model

METRICS_DIFFERENCE_THRESHOLD = 0.02

MODELS_NAMES_TO_CLASSES_MAPPING = {
    "xgboost": XGBClassifier,
    "random_forest": RandomForestClassifier,
}
MODELS_DEFAULT_PARAMS = {
    "xgboost": {"scale_pos_weight": 11.381618,},
    "random_forest": {
        "max_features": "sqrt",
        "n_jobs": -1,
        "random_state": 42,
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
    data_path: str = get_data_path(parameters["data_path"])

    logger.info("Creating optuna study")
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

    logger.info("Fitting model to train data")
    model.fit(X_train, y_train)

    logger.info("Evaluating model")
    logger.info(
        f"Metrics on train set: {get_metrics(model, X_train, y_train)}"
    )
    logger.info(
        f"Metrics on validation set: {get_metrics(model, X_valid, y_valid)}"
    )
    logger.info(f"Metrics on test set: {get_metrics(model, X_test, y_test)}")

    logger.info(f"Saving model")
    save_model(
        model,
        f"resources/models/serialized/{model_name}.pkl",
        f"resources/models/parameters/{model_name}.json",
    )


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
