import argparse
import os
import pandas as pd
from datetime import datetime
from typing import Callable, Dict
from functools import partial
from models.utils.save_model import save_model
from sklearn.model_selection import cross_validate
import optuna
import numpy as np
import utils.paths as paths
from models.utils.metrics import get_metrics
from utils.logging import optimization_logger


def optimize_model(default_model_name: str, define_model_fun: Callable):

    system_arguments = _get_system_arguments()
    model_name = (
        system_arguments["model_name"]
        if system_arguments["model_name"]
        else default_model_name
    )

    optimization_logger.info(
        f"Reading train data from {paths.DATA_TO_TRAIN_PATH}"
    )
    X_train = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "X_train.parquet")
    )
    y_train = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "y_train.parquet")
    )

    time = datetime.now().strftime("%Y-%m-%d-%H:%m:%S")
    study_filename = model_name + "-" + time
    storage_path = os.path.join(
        paths.OPTUNA_STUDIES_PATH, default_model_name, study_filename + ".db"
    )

    optimization_logger.info(f"Optuna storage path: {storage_path}")
    storage_name = f"sqlite:///{storage_path}"
    study_name = model_name
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage=storage_name,
        study_name=study_name,
    )
    study.optimize(
        partial(
            _objective, define_model_fun=define_model_fun, X=X_train, y=y_train
        ),
        n_jobs=-1,
        n_trials=system_arguments["n_trials"],
        timeout=system_arguments["timeout"],
    )

    trial_with_highest_f1 = max(
        study.best_trials, key=lambda trial: trial.values[1]
    )
    optimization_logger.info("Best model stats: ")
    optimization_logger.info(
        f"F1, F1_train - F1_test: {trial_with_highest_f1.values}"
    )

    X_test = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "X_test.parquet")
    )
    y_test = pd.read_parquet(
        os.path.join(paths.DATA_TO_TRAIN_PATH, "y_test.parquet")
    )

    model = define_model_fun(trial_with_highest_f1)
    model.fit(X_train, y_train)
    optimization_logger.info(
        f"Metrics on test dataset: {get_metrics(model, X_test, y_test)}"
    )

    if system_arguments["save_best"]:
        save_model(model, model_name)


def _objective(trial, define_model_fun, X, y):
    model = define_model_fun(trial)
    result = cross_validate(
        model, X, y, cv=3, scoring="f1", return_train_score=True
    )
    train_score = result["train_score"]
    test_score = result["test_score"]
    return np.mean(train_score), np.mean((train_score - test_score))


def _get_system_arguments() -> Dict[str, str]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_trials", help="Number of trials", default=None, type=int
    )
    parser.add_argument(
        "--timeout",
        help="Max time of optimization in seconds",
        default=None,
        type=int,
    )
    parser.add_argument(
        "--save_best",
        help="If set best model is saved as pickle and its parameters are persisted",
        action="store_true",
    )
    parser.add_argument(
        "--model_name", help="Used for naming files", type=str, required=False
    )
    args = parser.parse_args()
    return vars(args)
