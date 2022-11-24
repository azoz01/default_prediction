import logging
import argparse
from typing import Callable, Dict, Tuple
from functools import partial
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score
from utils.io import load_data, save_json
from utils.parameters import get_data_path, get_models_path
from utils.optuna import get_best_trial_params_cat

logger = logging.getLogger(__name__)

TRAIN_DATA = "reduced"


def optimize_model(
    default_model_name: str,
    define_model_fun: Callable = None,
    objective_fun: Callable = None,
) -> None:

    if define_model_fun is None and objective_fun is None:
        raise ValueError(
            "Either objective or define_function should be provided"
        )

    system_arguments = _get_system_arguments()
    model_name = (
        system_arguments["model_name"]
        if system_arguments["model_name"]
        else default_model_name
    )
    logger.info(f"Starting hyperparameter optimization for {model_name}")

    data_path = get_data_path(TRAIN_DATA)

    logger.info(f"Reading train data from {data_path}")

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

    study_filename = model_name
    optuna_studies_path = get_models_path("optuna_studies") / model_name

    if not optuna_studies_path.exists():
        optuna_studies_path.mkdir()
    storage_path = optuna_studies_path / (study_filename + ".db")
    if storage_path.exists():
        storage_path.unlink()
    storage_name = f"sqlite:///{storage_path}"
    logger.info(f"Optuna storage path: {storage_path}")

    study_name = model_name
    study = optuna.create_study(
        directions=["maximize", "minimize"],
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True,
    )
    study.optimize(
        objective_fun,
        n_jobs=-1,
        n_trials=system_arguments["n_trials"],
        timeout=system_arguments["timeout"],
    )
    best_trial_params = get_best_trial_params_cat(study)
    params_output_path = get_models_path("parameters") / (model_name + ".json")
    save_json(best_trial_params, params_output_path)


def _objective(
    trial,
    define_model_fun: Callable,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
) -> Tuple[float, float]:
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
