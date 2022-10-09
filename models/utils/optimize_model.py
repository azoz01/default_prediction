import os
import logging
import argparse
from datetime import datetime
from typing import Callable, Dict
from functools import partial
import optuna
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.io import load_data
from utils.parameters import get_data_path, get_models_path

logger = logging.getLogger(__name__)

TRAIN_DATA: str = "balanced_smote"


def optimize_model(
    default_model_name: str,
    define_model_fun: Callable = None,
    objective_fun: Callable = None,
):

    if define_model_fun is None and objective_fun is None:
        raise ValueError(
            "Either objective or define_function should be provided"
        )

    system_arguments: Dict[str, str] = _get_system_arguments()
    model_name: str = (
        system_arguments["model_name"]
        if system_arguments["model_name"]
        else default_model_name
    )
    logger.info(f"Starting hyperparameter optimization for {model_name}")

    data_path: str = get_data_path(TRAIN_DATA)

    logger.info(f"Reading train data from {data_path}")
    X_train: pd.DataFrame
    y_train: pd.DataFrame
    X_valid: pd.DataFrame
    y_valid: pd.DataFrame

    X_train, y_train, X_valid, y_valid, _, _ = load_data(path=data_path)

    if not objective_fun:
        objective_fun = _objective
        objective_fun = partial(
            objective_fun,
            define_model_fun=define_model_fun,
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
        )
    else:
        objective_fun = partial(objective_fun, X=X_train, y=y_train,)

    time: str = datetime.now().strftime("%Y-%m-%d-%H:%m:%S")
    study_filename: str = model_name + "-" + time
    optuna_studies_path: str = get_models_path("optuna_studies")
    optuna_studies_path: str = os.path.join(optuna_studies_path, model_name)
    if not os.path.exists(optuna_studies_path):
        os.mkdir(optuna_studies_path)
    storage_path: str = os.path.join(
        optuna_studies_path, study_filename + ".db"
    )
    storage_name: str = f"sqlite:///{storage_path}"
    logger.info(f"Optuna storage path: {storage_path}")

    study_name = model_name
    study: optuna.Study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage=storage_name,
        study_name=study_name,
    )
    study.optimize(
        objective_fun,
        n_jobs=-1,
        n_trials=system_arguments["n_trials"],
        timeout=system_arguments["timeout"],
    )


def _objective(trial, define_model_fun, X_train, y_train, X_valid, y_valid):
    model = define_model_fun(trial)
    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_valid = model.predict(X_valid)
    train_score = roc_auc_score(y_train, pred_train)
    valid_score = roc_auc_score(y_valid, pred_valid)
    return valid_score, train_score - valid_score


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
        "--model_name", help="Used for naming files", type=str, required=False
    )
    args = parser.parse_args()
    return vars(args)
